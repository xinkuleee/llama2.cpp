#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>

#include "win.h"
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace llama2 {
struct Config {
  std::int32_t dim;         // transformer dimension
  std::int32_t hidden_dim;  // for ffn layers
  std::int32_t n_layers;    // number of layers
  std::int32_t n_heads;     // number of query heads
  std::int32_t n_kv_heads;  // number of key/value heads (can be < query heads
                            // because of multiquery)
  std::int32_t vocab_size;  // vocabulary size, usually 256 (byte-level)
  std::int32_t seq_len;     // max sequence length

  void validate() const {
    if (dim <= 0 || hidden_dim <= 0 || n_layers <= 0 || n_heads <= 0 ||
        n_kv_heads <= 0 || vocab_size <= 0 || seq_len <= 0) {
      throw std::runtime_error("invalid Config: all fields must be positive");
    }

    if (dim % n_heads != 0) {
      throw std::runtime_error(
          "invalid Config: dim must be divisible by n_heads");
    }

    if (n_heads % n_kv_heads != 0) {
      throw std::runtime_error(
          "invalid Config: n_heads must be divisible by n_kv_heads");
    }

    if (head_size() % 2 != 0) {
      throw std::runtime_error(
          "invalid Config: head_size must be divisible by 2 for RoPE");
    }
  }

  std::size_t head_size() const noexcept {
    return static_cast<std::size_t>(dim) / static_cast<std::size_t>(n_heads);
  }

  std::size_t kv_dim() const noexcept {
    return static_cast<std::size_t>(dim) *
           static_cast<std::size_t>(n_kv_heads) /
           static_cast<std::size_t>(n_heads);
  }
};

static_assert(sizeof(Config) == 7 * sizeof(std::int32_t));

class MappedFile {
 public:
  explicit MappedFile(const std::filesystem::path& path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd == -1) {
      throw std::system_error(errno, std::generic_category(),
                              "failed to open " + path.string());
    }

    struct stat info{};
    if (::fstat(fd, &info) == -1) {
      const int error = errno;
      ::close(fd);
      throw std::system_error(error, std::generic_category(),
                              "failed to inspect " + path.string());
    }

    if (info.st_size <= 0) {
      ::close(fd);
      throw std::runtime_error("checkpoint is empty: " + path.string());
    }

    if (static_cast<std::uintmax_t>(info.st_size) >
        std::numeric_limits<std::size_t>::max()) {
      ::close(fd);
      throw std::runtime_error("checkpoint is too large");
    }

    size_ = static_cast<std::size_t>(info.st_size);

    void* mapping = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);

    const int mmap_error = errno;
    ::close(fd);

    if (mapping == MAP_FAILED) {
      size_ = 0;
      throw std::system_error(mmap_error, std::generic_category(),
                              "failed to map " + path.string());
    }

    mapping_ = mapping;
  }
  ~MappedFile() noexcept {
    if (mapping_ != MAP_FAILED) {
      ::munmap(mapping_, size_);
    }
  }

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  MappedFile(MappedFile&&) = delete;
  MappedFile& operator=(MappedFile&&) = delete;

  std::span<const std::byte> bytes() const noexcept {
    return {static_cast<const std::byte*>(mapping_), size_};
  }

  std::size_t size() const noexcept { return size_; }

 private:
  void* mapping_{MAP_FAILED};
  std::size_t size_{0};
};

// Non-owning, read-only view of Q8_0 values and their per-group scales.
struct QuantizedTensor {
  std::span<const std::int8_t> values;
  std::span<const float> scales;

  QuantizedTensor subspan(std::size_t offset, std::size_t count,
                          std::size_t group_size) const {
    if (group_size == 0 || offset % group_size != 0 ||
        count % group_size != 0 || offset > values.size() ||
        count > values.size() - offset ||
        scales.size() != values.size() / group_size) {
      throw std::invalid_argument("invalid quantized tensor subspan");
    }

    return {values.subspan(offset, count),
            scales.subspan(offset / group_size, count / group_size)};
  }
};

// Owning, writable storage for activations quantized during forward().
struct QuantizedBuffer {
  QuantizedBuffer() = default;

  QuantizedBuffer(std::size_t size, std::size_t group_size) : values(size) {
    if (group_size == 0 || size % group_size != 0) {
      throw std::invalid_argument("invalid quantized buffer size");
    }
    scales.resize(size / group_size);
  }

  QuantizedTensor view() const noexcept { return {values, scales}; }

  std::vector<std::int8_t> values;
  std::vector<float> scales;
};

class WeightCursor {
 public:
  explicit WeightCursor(std::span<const std::byte> payload) noexcept
      : payload_(payload) {}

  std::span<const float> take_fp32(std::size_t count,
                                   std::string_view weight_name) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
      throw std::runtime_error("FP32 tensor size overflow while reading " +
                               std::string{weight_name});
    }

    const auto bytes = take_bytes(count * sizeof(float), weight_name);
    const auto address = reinterpret_cast<std::uintptr_t>(bytes.data());
    if (address % alignof(float) != 0) {
      throw std::runtime_error("unaligned FP32 tensor while reading " +
                               std::string{weight_name});
    }

    return {reinterpret_cast<const float*>(bytes.data()), count};
  }

  QuantizedTensor take_q80(std::size_t count, std::size_t group_size,
                           std::string_view weight_name) {
    if (group_size == 0 || count % group_size != 0) {
      throw std::runtime_error("invalid Q8_0 tensor while reading " +
                               std::string{weight_name});
    }

    const auto value_bytes = take_bytes(count, weight_name);
    const auto scales = take_fp32(count / group_size, weight_name);
    return {{reinterpret_cast<const std::int8_t*>(value_bytes.data()), count},
            scales};
  }

  std::size_t remaining() const noexcept { return payload_.size() - position_; }

 private:
  std::span<const std::byte> take_bytes(std::size_t count,
                                        std::string_view weight_name) {
    if (count > remaining()) {
      throw std::runtime_error("checkpoint truncated while reading " +
                               std::string{weight_name} + ": requested " +
                               std::to_string(count) + ", remaining " +
                               std::to_string(remaining()));
    }

    auto result = payload_.subspan(position_, count);
    position_ += count;
    return result;
  }

  std::span<const std::byte> payload_;
  std::size_t position_{0};
};

struct TransformerWeights {
  std::span<const float> rms_att_weight;
  std::span<const float> rms_ffn_weight;
  std::span<const float> rms_final_weight;

  QuantizedTensor token_embedding_table;

  std::vector<QuantizedTensor> wq;
  std::vector<QuantizedTensor> wk;
  std::vector<QuantizedTensor> wv;
  std::vector<QuantizedTensor> wo;

  std::vector<QuantizedTensor> w1;
  std::vector<QuantizedTensor> w2;
  std::vector<QuantizedTensor> w3;

  QuantizedTensor wcls;
};

struct RunState {
  RunState() = default;
  RunState(const Config& config, std::size_t group_size)
      : x(static_cast<std::size_t>(config.dim)),
        xb(static_cast<std::size_t>(config.dim)),
        xb2(static_cast<std::size_t>(config.dim)),
        hb(static_cast<std::size_t>(config.hidden_dim)),
        hb2(static_cast<std::size_t>(config.hidden_dim)),
        xq(static_cast<std::size_t>(config.dim), group_size),
        hq(static_cast<std::size_t>(config.hidden_dim), group_size),
        q(static_cast<std::size_t>(config.dim)),
        attention(static_cast<std::size_t>(config.n_heads) *
                  static_cast<std::size_t>(config.seq_len)),
        logits(static_cast<std::size_t>(config.vocab_size)),
        key_cache(static_cast<std::size_t>(config.n_layers) *
                  static_cast<std::size_t>(config.seq_len) * config.kv_dim()),
        value_cache(static_cast<std::size_t>(config.n_layers) *
                    static_cast<std::size_t>(config.seq_len) *
                    config.kv_dim()) {}

  std::span<float> key_at(const Config& config, std::size_t layer,
                          std::size_t position) {
    const std::size_t kv_dim = config.kv_dim();
    const std::size_t seq_len = static_cast<std::size_t>(config.seq_len);
    return std::span<float>{key_cache}.subspan(
        (layer * seq_len + position) * kv_dim, kv_dim);
  }

  std::span<float> value_at(const Config& config, std::size_t layer,
                            std::size_t position) {
    const std::size_t kv_dim = config.kv_dim();
    const std::size_t seq_len = static_cast<std::size_t>(config.seq_len);
    return std::span<float>{value_cache}.subspan(
        (layer * seq_len + position) * kv_dim, kv_dim);
  }

  std::vector<float> x;
  std::vector<float> xb;
  std::vector<float> xb2;

  std::vector<float> hb;
  std::vector<float> hb2;

  QuantizedBuffer xq;
  QuantizedBuffer hq;

  std::vector<float> q;
  std::vector<float> attention;
  std::vector<float> logits;

  std::vector<float> key_cache;
  std::vector<float> value_cache;
};

namespace {
template <typename T>
T read_scalar(std::span<const std::byte> bytes, std::size_t offset,
              std::string_view field_name) {
  if (offset > bytes.size() || sizeof(T) > bytes.size() - offset) {
    throw std::runtime_error("checkpoint header is truncated at " +
                             std::string{field_name});
  }

  T value{};
  std::memcpy(&value, bytes.data() + offset, sizeof(T));
  return value;
}

void read_checkpoint(std::span<const std::byte> bytes, Config& config,
                     std::size_t& group_size, TransformerWeights& weights) {
  constexpr std::uint32_t expected_magic = 0x616b3432;
  constexpr std::int32_t expected_version = 2;
  constexpr std::size_t header_size = 256;
  constexpr std::size_t config_offset = 8;
  constexpr std::size_t shared_classifier_offset =
      config_offset + sizeof(Config);
  constexpr std::size_t group_size_offset = shared_classifier_offset + 1;

  if (bytes.size() < header_size) {
    throw std::runtime_error("checkpoint is smaller than its v2 header");
  }

  if (read_scalar<std::uint32_t>(bytes, 0, "magic") != expected_magic) {
    throw std::runtime_error("invalid checkpoint magic");
  }
  if (read_scalar<std::int32_t>(bytes, 4, "version") != expected_version) {
    throw std::runtime_error("runq.cpp requires a version 2 checkpoint");
  }

  std::memcpy(&config, bytes.data() + config_offset, sizeof(Config));
  config.validate();

  const std::uint8_t shared_classifier = read_scalar<std::uint8_t>(
      bytes, shared_classifier_offset, "shared_classifier");
  if (shared_classifier > 1) {
    throw std::runtime_error("invalid shared_classifier flag");
  }

  const std::int32_t checkpoint_group_size =
      read_scalar<std::int32_t>(bytes, group_size_offset, "group_size");
  if (checkpoint_group_size <= 0) {
    throw std::runtime_error("invalid quantization group size");
  }
  group_size = static_cast<std::size_t>(checkpoint_group_size);

  const std::size_t dim = static_cast<std::size_t>(config.dim);
  const std::size_t hidden_dim = static_cast<std::size_t>(config.hidden_dim);
  const std::size_t n_layers = static_cast<std::size_t>(config.n_layers);
  const std::size_t vocab_size = static_cast<std::size_t>(config.vocab_size);
  const std::size_t kv_dim = config.kv_dim();

  if (dim % group_size != 0 || hidden_dim % group_size != 0) {
    throw std::runtime_error(
        "model dimensions are incompatible with group size");
  }

  WeightCursor cursor{bytes.subspan(header_size)};

  weights.rms_att_weight = cursor.take_fp32(n_layers * dim, "rms_att_weight");
  weights.rms_ffn_weight = cursor.take_fp32(n_layers * dim, "rms_ffn_weight");
  weights.rms_final_weight = cursor.take_fp32(dim, "rms_final_weight");

  weights.token_embedding_table =
      cursor.take_q80(vocab_size * dim, group_size, "token_embedding_table");

  const auto take_layers = [&](std::vector<QuantizedTensor>& tensors,
                               std::size_t element_count,
                               std::string_view name) {
    tensors.reserve(n_layers);
    for (std::size_t layer = 0; layer < n_layers; ++layer) {
      tensors.push_back(cursor.take_q80(element_count, group_size, name));
    }
  };

  take_layers(weights.wq, dim * dim, "wq");
  take_layers(weights.wk, kv_dim * dim, "wk");
  take_layers(weights.wv, kv_dim * dim, "wv");
  take_layers(weights.wo, dim * dim, "wo");
  take_layers(weights.w1, hidden_dim * dim, "w1");
  take_layers(weights.w2, dim * hidden_dim, "w2");
  take_layers(weights.w3, hidden_dim * dim, "w3");

  weights.wcls = shared_classifier
                     ? weights.token_embedding_table
                     : cursor.take_q80(vocab_size * dim, group_size, "wcls");
}

int hex_digit(char value) noexcept {
  if (value >= '0' && value <= '9') {
    return value - '0';
  }
  if (value >= 'a' && value <= 'f') {
    return value - 'a' + 10;
  }
  if (value >= 'A' && value <= 'F') {
    return value - 'A' + 10;
  }
  return -1;
}

void safe_print(std::string_view piece) {
  if (piece.empty()) {
    return;
  }
  if (piece.size() == 1) {
    const auto byte = static_cast<unsigned char>(piece.front());
    if (std::isprint(byte) == 0 && std::isspace(byte) == 0) {
      return;
    }
  }
  std::cout.write(piece.data(), static_cast<std::streamsize>(piece.size()));
}

std::string read_line(std::string_view prompt) {
  std::cout << prompt << std::flush;

  std::string line;

  if (!std::getline(std::cin, line)) {
    throw std::runtime_error("failed to read standard input");
  }

  return line;
}

std::int64_t parse_integer(std::string_view text, std::string_view option) {
  std::int64_t value{};

  const char* begin = text.data();
  const char* end = begin + text.size();

  const auto result = std::from_chars(begin, end, value);

  if (result.ec != std::errc{} || result.ptr != end) {
    throw std::invalid_argument("invalid value for " + std::string{option} +
                                ": " + std::string{text});
  }

  return value;
}

float parse_float(std::string_view text, std::string_view option) {
  std::string owned{text};
  char* end = nullptr;
  errno = 0;

  const float value = std::strtof(owned.c_str(), &end);
  if (end == owned.c_str() || end != owned.c_str() + owned.size() ||
      errno == ERANGE || !std::isfinite(value)) {
    throw std::invalid_argument("invalid value for " + std::string{option} +
                                ": " + owned);
  }

  return value;
}
}  // namespace

class Transformer {
 public:
  explicit Transformer(const std::filesystem::path& checkpoint_path)
      : checkpoint_(checkpoint_path) {
    read_checkpoint(checkpoint_.bytes(), config_, group_size_, weights_);
    state_ = RunState(config_, group_size_);
  }

  const Config& config() const noexcept { return config_; }

  std::span<float> forward(std::int32_t token, std::size_t position);

 private:
  MappedFile checkpoint_;
  Config config_;
  std::size_t group_size_{0};
  TransformerWeights weights_;
  RunState state_;
};

namespace kernel {
void rms_norm(std::span<float> output, std::span<const float> input,
              std::span<const float> weight) {
  if (input.empty() || input.size() != output.size() ||
      input.size() != weight.size()) {
    throw std::runtime_error(
        "input, output, and weight must have the same non-zero size");
  }

  float sum_of_squares = 0.0F;
  for (const float value : input) {
    sum_of_squares += value * value;
  }

  const float mean_square = sum_of_squares / static_cast<float>(input.size());
  const float inverse_rms = 1.0F / std::sqrt(mean_square + 1e-5F);

  for (std::size_t index = 0; index < input.size(); index++) {
    output[index] = weight[index] * input[index] * inverse_rms;
  }
}

void softmax_inplace(std::span<float> values) {
  if (values.empty()) {
    throw std::invalid_argument("softmax input must not be empty");
  }

  const float maximum = *std::max_element(values.begin(), values.end());
  float exponential_sum = 0.0F;

  for (float& value : values) {
    value = std::exp(value - maximum);
    exponential_sum += value;
  }

  for (float& value : values) {
    value /= exponential_sum;
  }
}

void quantize_q80(QuantizedBuffer& output, std::span<const float> input,
                  std::size_t group_size) {
  if (group_size == 0 || input.size() % group_size != 0 ||
      output.values.size() != input.size() ||
      output.scales.size() != input.size() / group_size) {
    throw std::invalid_argument("Q8_0 quantization size mismatch");
  }

  for (std::size_t group_begin = 0; group_begin < input.size();
       group_begin += group_size) {
    float maximum = 0.0F;
    for (std::size_t index = 0; index < group_size; ++index) {
      maximum = std::max(maximum, std::abs(input[group_begin + index]));
    }

    const std::size_t group = group_begin / group_size;
    if (maximum == 0.0F) {
      output.scales[group] = 0.0F;
      std::fill_n(
          output.values.begin() + static_cast<std::ptrdiff_t>(group_begin),
          group_size, std::int8_t{0});
      continue;
    }

    const float scale = maximum / 127.0F;
    output.scales[group] = scale;

    for (std::size_t index = 0; index < group_size; ++index) {
      const float quantized = std::round(input[group_begin + index] / scale);
      output.values[group_begin + index] =
          static_cast<std::int8_t>(std::clamp(quantized, -127.0F, 127.0F));
    }
  }
}

void dequantize_q80(std::span<float> output, const QuantizedTensor& input,
                    std::size_t group_size) {
  if (group_size == 0 || input.values.size() % group_size != 0 ||
      output.size() != input.values.size() ||
      input.scales.size() != input.values.size() / group_size) {
    throw std::invalid_argument("Q8_0 dequantization size mismatch");
  }

  for (std::size_t index = 0; index < output.size(); ++index) {
    output[index] = static_cast<float>(input.values[index]) *
                    input.scales[index / group_size];
  }
}

void matmul_q80(std::span<float> output, const QuantizedTensor& matrix,
                const QuantizedTensor& input, std::size_t rows,
                std::size_t columns, std::size_t group_size) {
  if (rows == 0 || columns == 0 || group_size == 0 ||
      columns % group_size != 0 ||
      rows > std::numeric_limits<std::size_t>::max() / columns) {
    throw std::invalid_argument("invalid Q8_0 matmul dimensions");
  }

  const std::size_t matrix_size = rows * columns;
  if (output.size() != rows || input.values.size() != columns ||
      input.scales.size() != columns / group_size ||
      matrix.values.size() != matrix_size ||
      matrix.scales.size() != matrix_size / group_size) {
    throw std::invalid_argument("Q8_0 matmul size mismatch");
  }

  for (std::size_t row = 0; row < rows; ++row) {
    float result = 0.0F;
    const std::size_t row_begin = row * columns;

    for (std::size_t group_begin = 0; group_begin < columns;
         group_begin += group_size) {
      std::int32_t integer_dot = 0;
      for (std::size_t index = 0; index < group_size; ++index) {
        integer_dot +=
            static_cast<std::int32_t>(input.values[group_begin + index]) *
            static_cast<std::int32_t>(
                matrix.values[row_begin + group_begin + index]);
      }

      result += static_cast<float>(integer_dot) *
                matrix.scales[(row_begin + group_begin) / group_size] *
                input.scales[group_begin / group_size];
    }

    output[row] = result;
  }
}

void apply_rope(std::span<float> query, std::span<float> key,
                std::size_t position, std::size_t head_size) {
  if (head_size == 0 || head_size % 2 != 0) {
    throw std::runtime_error("head_size must be positive and even");
  }
  if (query.size() % head_size != 0 || key.size() % head_size != 0) {
    throw std::runtime_error(
        "query and key must have size divisible by head_size");
  }
  const auto rotate = [position, head_size](std::span<float> values) {
    for (std::size_t head_begin = 0; head_begin < values.size();
         head_begin += head_size) {
      for (std::size_t offset = 0; offset < head_size; offset += 2) {
        const float frequency =
            1.0F / std::pow(10000.0F, static_cast<float>(offset) /
                                          static_cast<float>(head_size));

        const float angle = static_cast<float>(position) * frequency;
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);

        const std::size_t first_index = head_begin + offset;
        const std::size_t second_index = head_begin + offset + 1;

        const float first_value = values[first_index];
        const float second_value = values[second_index];

        values[first_index] = first_value * cosine - second_value * sine;
        values[second_index] = first_value * sine + second_value * cosine;
      }
    }
  };
  rotate(query);
  rotate(key);
}
}  // namespace kernel

std::span<float> Transformer::forward(std::int32_t token,
                                      std::size_t position) {
  const std::size_t dim = static_cast<std::size_t>(config_.dim);
  const std::size_t hidden_dim = static_cast<std::size_t>(config_.hidden_dim);
  const std::size_t n_layers = static_cast<std::size_t>(config_.n_layers);
  const std::size_t n_heads = static_cast<std::size_t>(config_.n_heads);
  const std::size_t n_kv_heads = static_cast<std::size_t>(config_.n_kv_heads);
  const std::size_t seq_len = static_cast<std::size_t>(config_.seq_len);
  const std::size_t vocab_size = static_cast<std::size_t>(config_.vocab_size);
  const std::size_t head_size = config_.head_size();
  const std::size_t kv_dim = config_.kv_dim();
  const std::size_t kv_mul = n_heads / n_kv_heads;

  if (token < 0 || static_cast<std::size_t>(token) >= vocab_size) {
    throw std::runtime_error("token out of range");
  }
  if (position >= seq_len) {
    throw std::runtime_error("position out of range");
  }

  auto x = std::span<float>{state_.x};
  auto xb = std::span<float>{state_.xb};
  auto xb2 = std::span<float>{state_.xb2};
  auto q = std::span<float>{state_.q};
  auto hb = std::span<float>{state_.hb};
  auto hb2 = std::span<float>{state_.hb2};
  auto logits = std::span<float>{state_.logits};

  const QuantizedTensor embedding = weights_.token_embedding_table.subspan(
      static_cast<std::size_t>(token) * dim, dim, group_size_);
  kernel::dequantize_q80(x, embedding, group_size_);

  const float score_scale = 1.0F / std::sqrt(static_cast<float>(head_size));

  for (std::size_t layer = 0; layer < n_layers; layer++) {
    // ----- Attention -----
    kernel::rms_norm(xb, x, weights_.rms_att_weight.subspan(layer * dim, dim));

    auto current_key = state_.key_at(config_, layer, position);
    auto current_value = state_.value_at(config_, layer, position);

    kernel::quantize_q80(state_.xq, xb, group_size_);
    kernel::matmul_q80(q, weights_.wq[layer], state_.xq.view(), dim, dim,
                       group_size_);
    kernel::matmul_q80(current_key, weights_.wk[layer], state_.xq.view(),
                       kv_dim, dim, group_size_);
    kernel::matmul_q80(current_value, weights_.wv[layer], state_.xq.view(),
                       kv_dim, dim, group_size_);

    kernel::apply_rope(q, current_key, position, head_size);

    for (std::size_t head = 0; head < n_heads; head++) {
      const auto query_head =
          std::span<const float>{q}.subspan(head * head_size, head_size);
      auto attention = std::span<float>{state_.attention}.subspan(
          head * seq_len, position + 1);

      const std::size_t kv_head = head / kv_mul;
      const std::size_t kv_head_begin = kv_head * head_size;

      for (std::size_t timestep = 0; timestep <= position; ++timestep) {
        const std::span<const float> history_key =
            state_.key_at(config_, layer, timestep)
                .subspan(kv_head_begin, head_size);

        float score = 0.0F;
        for (std::size_t index = 0; index < head_size; ++index) {
          score += query_head[index] * history_key[index];
        }
        attention[timestep] = score * score_scale;
      }

      kernel::softmax_inplace(attention);

      auto head_output = xb.subspan(head * head_size, head_size);
      std::fill(head_output.begin(), head_output.end(), 0.0F);

      for (std::size_t timestep = 0; timestep <= position; ++timestep) {
        const std::span<const float> history_value =
            state_.value_at(config_, layer, timestep)
                .subspan(kv_head_begin, head_size);

        const float attention_weight = attention[timestep];
        for (std::size_t index = 0; index < head_size; ++index) {
          head_output[index] += attention_weight * history_value[index];
        }
      }
    }

    kernel::quantize_q80(state_.xq, xb, group_size_);
    kernel::matmul_q80(xb2, weights_.wo[layer], state_.xq.view(), dim, dim,
                       group_size_);

    for (std::size_t index = 0; index < dim; ++index) {
      x[index] += xb2[index];
    }

    // ----- FFN -----
    kernel::rms_norm(xb, x, weights_.rms_ffn_weight.subspan(layer * dim, dim));

    kernel::quantize_q80(state_.xq, xb, group_size_);
    kernel::matmul_q80(hb, weights_.w1[layer], state_.xq.view(), hidden_dim,
                       dim, group_size_);
    kernel::matmul_q80(hb2, weights_.w3[layer], state_.xq.view(), hidden_dim,
                       dim, group_size_);

    for (std::size_t index = 0; index < hidden_dim; ++index) {
      const float gate = hb[index];
      const float silu = gate / (1.0F + std::exp(-gate));
      hb[index] = silu * hb2[index];
    }

    kernel::quantize_q80(state_.hq, hb, group_size_);
    kernel::matmul_q80(xb, weights_.w2[layer], state_.hq.view(), dim,
                       hidden_dim, group_size_);

    for (std::size_t index = 0; index < dim; ++index) {
      x[index] += xb[index];
    }
  }

  kernel::rms_norm(x, x, weights_.rms_final_weight);
  kernel::quantize_q80(state_.xq, x, group_size_);
  kernel::matmul_q80(logits, weights_.wcls, state_.xq.view(), vocab_size, dim,
                     group_size_);
  return logits;
}

class Tokenizer {
 public:
  Tokenizer(const std::filesystem::path& tokenizer_path,
            std::size_t vocab_size) {
    std::ifstream input(tokenizer_path, std::ios::binary);

    if (!input) {
      throw std::runtime_error("failed to open tokenizer: " +
                               tokenizer_path.string());
    }

    std::uint32_t max_token_length = 0;
    input.read(reinterpret_cast<char*>(&max_token_length),
               sizeof(max_token_length));

    if (!input || max_token_length == 0) {
      throw std::runtime_error("invalid tokenizer header");
    }

    max_token_length_ = max_token_length;
    vocabulary_.resize(vocab_size);
    scores_.resize(vocab_size);
    sorted_token_ids_.resize(vocab_size);

    for (std::size_t token_id = 0; token_id < vocab_size; token_id++) {
      std::uint32_t piece_length = 0;

      input.read(reinterpret_cast<char*>(&scores_[token_id]), sizeof(float));

      input.read(reinterpret_cast<char*>(&piece_length), sizeof(piece_length));

      if (!input || piece_length > max_token_length_) {
        throw std::runtime_error("invalid tokenizer entry");
      }

      vocabulary_[token_id].resize(piece_length);

      input.read(vocabulary_[token_id].data(),
                 static_cast<std::streamsize>(piece_length));

      if (!input) {
        throw std::runtime_error("truncated tokenizer file");
      }

      sorted_token_ids_[token_id] = token_id;
    }

    std::sort(sorted_token_ids_.begin(), sorted_token_ids_.end(),
              [this](std::size_t left, std::size_t right) {
                return vocabulary_[left] < vocabulary_[right];
              });

    for (std::size_t byte = 0; byte < 256; ++byte) {
      byte_pieces_[byte][0] = static_cast<char>(byte);
      byte_pieces_[byte][1] = '\0';
    }
  }

  std::vector<std::int32_t> encode(std::string_view text, bool add_bos,
                                   bool add_eos) const;

  std::string_view decode(std::int32_t previous_token,
                          std::int32_t token) const;

 private:
  std::optional<std::int32_t> lookup(std::string_view piece) const {
    const auto iterator = std::lower_bound(
        sorted_token_ids_.begin(), sorted_token_ids_.end(), piece,
        [this](std::size_t token_id, std::string_view value) {
          return vocabulary_[token_id] < value;
        });

    if (iterator == sorted_token_ids_.end() ||
        vocabulary_[*iterator] != piece) {
      return std::nullopt;
    }

    return static_cast<std::int32_t>(*iterator);
  }

  std::vector<std::string> vocabulary_;
  std::vector<float> scores_;
  std::vector<std::size_t> sorted_token_ids_;
  std::array<std::array<char, 2>, 256> byte_pieces_{};

  std::size_t max_token_length_{0};
};

std::vector<std::int32_t> Tokenizer::encode(std::string_view text, bool add_bos,
                                            bool add_eos) const {
  if (vocabulary_.size() < 3) {
    throw std::runtime_error("tokenizer vocabulary is too small");
  }

  std::vector<std::int32_t> tokens;
  tokens.reserve(text.size() + 3);

  if (add_bos) {
    tokens.push_back(1);
  }

  // SentencePiece 默认在非空文本前增加空格。
  if (!text.empty()) {
    const auto prefix = lookup(" ");

    if (!prefix) {
      throw std::runtime_error("tokenizer has no dummy-prefix token");
    }

    tokens.push_back(*prefix);
  }

  // 先把 UTF-8 文本拆成 Unicode code point。
  std::string codepoint;
  codepoint.reserve(4);

  for (std::size_t index = 0; index < text.size(); ++index) {
    const auto byte = static_cast<unsigned char>(text[index]);

    // 非 continuation byte 表示新的 code point。
    if ((byte & 0xC0U) != 0x80U) {
      codepoint.clear();
    }

    codepoint.push_back(text[index]);

    const bool next_is_continuation =
        index + 1 < text.size() &&
        (static_cast<unsigned char>(text[index + 1]) & 0xC0U) == 0x80U;

    if (next_is_continuation && codepoint.size() < 4) {
      continue;
    }

    if (const auto token = lookup(codepoint)) {
      tokens.push_back(*token);
    } else {
      // 词表没有整个 code point 时退化为逐字节 token。
      for (const char raw_character : codepoint) {
        const auto raw_byte = static_cast<unsigned char>(raw_character);

        const std::size_t fallback_token =
            static_cast<std::size_t>(raw_byte) + 3;

        if (fallback_token >= vocabulary_.size()) {
          throw std::runtime_error("byte-fallback token is outside vocabulary");
        }

        tokens.push_back(static_cast<std::int32_t>(fallback_token));
      }
    }

    codepoint.clear();
  }

  // 反复合并当前分数最高的相邻 token。
  std::string merged_piece;
  merged_piece.reserve(max_token_length_ * 2);

  while (tokens.size() >= 2) {
    float best_score = std::numeric_limits<float>::lowest();

    std::size_t best_position = tokens.size();
    std::int32_t best_token = -1;

    for (std::size_t index = 0; index + 1 < tokens.size(); ++index) {
      merged_piece.clear();

      merged_piece += vocabulary_[static_cast<std::size_t>(tokens[index])];

      merged_piece += vocabulary_[static_cast<std::size_t>(tokens[index + 1])];

      const auto candidate = lookup(merged_piece);

      if (candidate &&
          scores_[static_cast<std::size_t>(*candidate)] > best_score) {
        best_score = scores_[static_cast<std::size_t>(*candidate)];

        best_position = index;
        best_token = *candidate;
      }
    }

    if (best_position == tokens.size()) {
      break;
    }

    tokens[best_position] = best_token;

    tokens.erase(tokens.begin() +
                 static_cast<std::ptrdiff_t>(best_position + 1));
  }

  if (add_eos) {
    tokens.push_back(2);
  }

  return tokens;
}

std::string_view Tokenizer::decode(std::int32_t previous_token,
                                   std::int32_t token) const {
  if (token < 0 || static_cast<std::size_t>(token) >= vocabulary_.size()) {
    throw std::out_of_range("decode token out of range");
  }

  std::string_view piece = vocabulary_[static_cast<std::size_t>(token)];

  // BOS 后移除 SentencePiece 添加的 dummy prefix。
  if (previous_token == 1 && !piece.empty() && piece.front() == ' ') {
    piece.remove_prefix(1);
  }

  // 把 "<0x41>" 这种 byte token 恢复为原始字节。
  if (piece.size() == 6 && piece[0] == '<' && piece[1] == '0' &&
      piece[2] == 'x' && piece[5] == '>') {
    const int high = hex_digit(piece[3]);
    const int low = hex_digit(piece[4]);

    if (high >= 0 && low >= 0) {
      const std::size_t byte = static_cast<std::size_t>((high << 4) | low);

      return {byte_pieces_[byte].data(), 1};
    }
  }

  return piece;
}

struct ProbIndex {
  float prob{};
  std::int32_t index{};
};  // struct used when sorting probabilities during top-p sampling

class Sampler {
 public:
  Sampler(std::size_t vocab_size, float temperature, float top_p,
          std::uint64_t seed)
      : vocab_size_(vocab_size),
        temperature_(temperature),
        top_p_(top_p),
        rng_state_(seed),
        scratch_(vocab_size) {
    if (vocab_size == 0 ||
        vocab_size > static_cast<std::size_t>(
                         std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument("invalid sampler vocabulary size");
    }

    if (!std::isfinite(temperature) || temperature < 0.0F) {
      throw std::invalid_argument(
          "temperature must be finite and non-negative");
    }

    if (!std::isfinite(top_p) || top_p < 0.0F || top_p > 1.0F) {
      throw std::invalid_argument("top_p must be in [0, 1]");
    }

    // xorshift64* must never start from its absorbing all-zero state.
    while (rng_state_ == 0) {
      rng_state_ = std::random_device{}();
    }
  }

  std::int32_t sample(std::span<float> logits) {
    if (logits.size() != vocab_size_) {
      throw std::invalid_argument("logits size does not match vocabulary");
    }
    if (temperature_ == 0.0F) {
      return sample_argmax(logits);
    }

    for (float& logit : logits) {
      logit /= temperature_;
    }

    kernel::softmax_inplace(logits);
    const float coin = random_f32();

    if (top_p_ <= 0.0F || top_p_ >= 1.0F) {
      return sample_mult(logits, coin);
    }

    return sample_topp(logits, coin);
  }

 private:
  std::int32_t sample_argmax(std::span<const float> values) const {
    if (values.empty()) {
      throw std::invalid_argument("empty sampling input");
    }

    std::size_t best = 0;
    for (std::size_t index = 1; index < values.size(); ++index) {
      if (values[index] > values[best]) {
        best = index;
      }
    }

    return static_cast<std::int32_t>(best);
  }

  std::int32_t sample_mult(std::span<const float> probabilities,
                           float coin) const {
    if (probabilities.empty()) {
      throw std::invalid_argument("empty probability distribution");
    }

    float cumulative = 0.0F;
    for (std::size_t index = 0; index < probabilities.size(); ++index) {
      cumulative += probabilities[index];

      if (coin < cumulative) {
        return static_cast<std::int32_t>(index);
      }
    }
    // 处理浮点误差导致累计和略小于 1 的情况。
    return static_cast<std::int32_t>(probabilities.size() - 1);
  }

  std::int32_t sample_topp(std::span<const float> probabilities, float coin) {
    if (probabilities.size() == 1) {
      return 0;
    }

    const float cutoff =
        (1.0F - top_p_) / static_cast<float>(probabilities.size() - 1);

    std::size_t candidate_count = 0;

    for (std::size_t token = 0; token < probabilities.size(); ++token) {
      if (probabilities[token] >= cutoff) {
        scratch_[candidate_count++] = {probabilities[token],
                                       static_cast<std::int32_t>(token)};
      }
    }

    if (candidate_count == 0) {
      return sample_argmax(probabilities);
    }

    std::sort(scratch_.begin(),
              scratch_.begin() + static_cast<std::ptrdiff_t>(candidate_count),
              [](const ProbIndex& left, const ProbIndex& right) {
                return left.prob > right.prob;
              });

    float nucleus_mass = 0.0F;
    std::size_t last_candidate = candidate_count - 1;

    for (std::size_t index = 0; index < candidate_count; ++index) {
      nucleus_mass += scratch_[index].prob;

      if (nucleus_mass > top_p_) {
        last_candidate = index;
        break;
      }
    }

    const float target = coin * nucleus_mass;
    float cumulative = 0.0F;

    for (std::size_t index = 0; index <= last_candidate; ++index) {
      cumulative += scratch_[index].prob;

      if (target < cumulative) {
        return scratch_[index].index;
      }
    }

    return scratch_[last_candidate].index;
  }

  std::uint32_t random_u32() {
    rng_state_ ^= rng_state_ >> 12;
    rng_state_ ^= rng_state_ << 25;
    rng_state_ ^= rng_state_ >> 27;
    return static_cast<std::uint32_t>((rng_state_ * 0x2545F4914F6CDD1DULL) >>
                                      32);
  }

  float random_f32() {
    return static_cast<float>(random_u32() >> 8) / 16777216.0F;
  }

  std::size_t vocab_size_{};
  float temperature_{};
  float top_p_{};
  std::uint64_t rng_state_{};

  std::vector<ProbIndex> scratch_;
};

enum class Mode { Generate, Chat };

struct Options {
  std::filesystem::path checkpoint_path;
  std::filesystem::path tokenizer_path{"tokenizer.bin"};

  float temperature{1.0F};
  float top_p{0.9F};
  std::uint64_t seed{0};
  std::size_t steps{256};

  std::string prompt;
  std::string system_prompt;
  Mode mode{Mode::Generate};
};

Options parse_options(int argc, char** argv) {
  if (argc < 2) {
    throw std::invalid_argument(
        "usage: runq_cpp <checkpoint> "
        "[-t temperature] [-p top_p] "
        "[-s seed] [-n steps] "
        "[-i prompt] [-z tokenizer] "
        "[-m generate|chat] "
        "[-y system_prompt]");
  }

  Options options;
  options.checkpoint_path = argv[1];

  for (int index = 2; index < argc; index += 2) {
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value for option " +
                                  std::string{argv[index]});
    }

    const std::string_view flag = argv[index];
    const std::string_view value = argv[index + 1];

    if (flag == "-t") {
      options.temperature = parse_float(value, flag);
    } else if (flag == "-p") {
      options.top_p = parse_float(value, flag);
    } else if (flag == "-s") {
      const std::int64_t seed = parse_integer(value, flag);

      options.seed = seed > 0 ? static_cast<std::uint64_t>(seed) : 0;
    } else if (flag == "-n") {
      const std::int64_t steps = parse_integer(value, flag);

      options.steps = steps > 0 ? static_cast<std::size_t>(steps) : 0;
    } else if (flag == "-i") {
      options.prompt = value;
    } else if (flag == "-z") {
      options.tokenizer_path = value;
    } else if (flag == "-m") {
      if (value == "generate") {
        options.mode = Mode::Generate;
      } else if (value == "chat") {
        options.mode = Mode::Chat;
      } else {
        throw std::invalid_argument("mode must be generate or chat");
      }
    } else if (flag == "-y") {
      options.system_prompt = value;
    } else {
      throw std::invalid_argument("unknown option: " + std::string{flag});
    }
  }

  // 与 run.c 的参数修正规则一致。
  if (options.temperature < 0.0F) {
    options.temperature = 0.0F;
  }
  if (options.top_p < 0.0F || options.top_p > 1.0F) {
    options.top_p = 0.9F;
  }
  return options;
}

void generate(Transformer& model, const Tokenizer& tokenizer, Sampler& sampler,
              std::string_view prompt, std::size_t steps) {
  const std::vector<std::int32_t> prompt_tokens =
      tokenizer.encode(prompt, true, false);

  if (prompt_tokens.empty()) {
    throw std::runtime_error("prompt encoding produced no tokens");
  }

  std::int32_t token = prompt_tokens.front();
  std::size_t position = 0;

  std::optional<std::chrono::steady_clock::time_point> start;
  while (position < steps) {
    std::span<float> logits = model.forward(token, position);

    const std::int32_t next_token = position + 1 < prompt_tokens.size()
                                        ? prompt_tokens[position + 1]
                                        : sampler.sample(logits);

    position++;
    // 与 run.c 一致：BOS 作为生成结束标记。
    if (next_token == 1) {
      break;
    }

    safe_print(tokenizer.decode(token, next_token));
    std::cout << std::flush;
    token = next_token;
    // 第一轮可能包含初始化开销，因此从其后计时。
    if (!start) {
      start = std::chrono::steady_clock::now();
    }
  }
  std::cout << '\n';
  if (position > 1 && start) {
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - *start)
            .count();
    if (elapsed > 0.0) {
      std::cerr << "achieved tok/s: "
                << static_cast<double>(position - 1) / elapsed << '\n';
    }
  }
}

void chat(Transformer& model, const Tokenizer& tokenizer, Sampler& sampler,
          std::string_view initial_prompt, std::string_view system_prompt,
          std::size_t steps) {
  std::string system{system_prompt};

  std::vector<std::int32_t> prompt_tokens;
  std::size_t prompt_index = 0;

  std::int32_t next_token = 0;
  bool have_next_token = false;
  bool need_user_prompt = true;

  for (std::size_t position = 0; position < steps; ++position) {
    if (need_user_prompt) {
      if (position == 0 && system.empty()) {
        system = read_line("Enter system prompt (optional): ");
      }

      const std::string user_prompt = position == 0 && !initial_prompt.empty()
                                          ? std::string{initial_prompt}
                                          : read_line("User: ");
      std::string rendered_prompt;

      if (position == 0 && !system.empty()) {
        rendered_prompt = "[INST] <<SYS>>\n" + system + "\n<</SYS>>\n\n" +
                          user_prompt + " [/INST]";
      } else {
        rendered_prompt = "[INST] " + user_prompt + " [/INST]";
      }

      prompt_tokens = tokenizer.encode(rendered_prompt, true, false);
      if (prompt_tokens.empty()) {
        throw std::runtime_error("chat prompt encoding produced no tokens");
      }

      prompt_index = 0;
      need_user_prompt = false;

      std::cout << "Assistant: " << std::flush;
    }

    std::int32_t token;
    if (prompt_index < prompt_tokens.size()) {
      token = prompt_tokens[prompt_index++];
    } else {
      if (!have_next_token) {
        throw std::logic_error("chat has no sampled token");
      }
      token = next_token;
    }

    // EOS 仍然需要喂给模型，让它进入 KV Cache，
    // 但其后产生的临时预测不应打印。
    const bool feeding_eos = token == 2;

    next_token = sampler.sample(model.forward(token, position));
    have_next_token = true;

    if (feeding_eos) {
      need_user_prompt = true;
      continue;
    }
    // prompt 消费完之后才输出 assistant token。
    if (prompt_index >= prompt_tokens.size()) {
      if (next_token == 2) {
        std::cout << '\n';
      } else {
        safe_print(tokenizer.decode(token, next_token));

        std::cout << std::flush;
      }
    }
  }
  std::cout << '\n';
}
}  // namespace llama2

using namespace llama2;
using namespace llama2::kernel;

int main(int argc, char** argv) {
  try {
    Options options = parse_options(argc, argv);
    Transformer model(options.checkpoint_path);

    const std::size_t sequence_limit =
        static_cast<std::size_t>(model.config().seq_len);
    if (options.steps == 0 || options.steps > sequence_limit) {
      options.steps = sequence_limit;
    }

    const std::size_t vocab_size =
        static_cast<std::size_t>(model.config().vocab_size);
    Tokenizer tokenizer(options.tokenizer_path, vocab_size);
    Sampler sampler(vocab_size, options.temperature, options.top_p,
                    options.seed);

    if (options.mode == Mode::Generate) {
      generate(model, tokenizer, sampler, options.prompt, options.steps);
    } else {
      chat(model, tokenizer, sampler, options.prompt, options.system_prompt,
           options.steps);
    }

    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
