#pragma once

#include <cstdint>
#include <memory>
#include <system_error>
#include <tl/expected.hpp>

namespace dh {
/**
 * @brief Enum class representing errors that can occur when manipulating
 * Holofile files.
 *
 * @note This error enum is not exhaustive and may be extended in the future.
 *
 * @note This error enum is not exhaustive as other errors may be returned by
 * the underlying file I/O operations.
 */
enum class HolofileError {
  /**
   * @brief Indicates that the end of the file has been reached.
   */
  EndOfFile = 1,

  /**
   * @brief Indicates that the header is incomplete.
   *
   * This can occur if the file does not contain enough bytes to read the
   * header.
   */
  IncompleteHeader,

  /**
   * @brief Indicates that the magic number is invalid.
   *
   * The magic number must be 0x484F4C4F or 0x4F4C4F48 depending on the
   * endianness, which corresponds to the ASCII string "HOLO".
   */
  InvalidMagicNumber,

  /**
   * @brief Indicates that the version number is invalid.
   *
   * The current supported version is 2.
   */
  InvalidVersion,

  /**
   * @brief Indicates that the frame size is invalid.
   *
   * The frame size must be a multiple of 8 bits.
   */
  InvalidFrameSize,
};

/**
 * @brief Error category for Holofile errors.
 *
 * This class provides the error category for Holofile errors. It is used to
 * provide custom error messages for Holofile errors.
 */
/**
 * @brief Represents an error category for holofile.
 *
 * This class inherits from std::error_category and provides a custom error
 * category for holofile.
 */
class holofile_error_category : public std::error_category {
public:
  /**
   * @brief Get the name of the error category.
   *
   * @return A pointer to a C-style string containing the name of the error
   * category.
   */
  const char *name() const noexcept override;

  /**
   * @brief Get the error message associated with the given error value.
   *
   * @param ev The error value.
   * @return A string containing the error message.
   */
  std::string message(int ev) const override;
};

/**
 * Returns the error category for holofile.
 *
 * @return A reference to the std::error_category object representing the error
 * category for holofile.
 */
const std::error_category &holofile_category();

/**
 * @brief Creates an error code object for the specified HolofileError.
 *
 * @param e The HolofileError value to create an error code for.
 * @return An std::error_code object representing the specified HolofileError.
 */
std::error_code make_error_code(HolofileError e);

#pragma pack(push, 1)
/**
 * @brief Header structure for the Holofile format.
 *
 * The Holofile format is a simple file format for storing a sequence of
 * black and white frames. The header structure contains metadata about the
 * file, such as the magic number, version number, frame size, and frame count.
 */
struct HolofileHeader {
  /**
   * @brief Magic number to identify the file format.
   *
   * The magic number is used to identify the file format. It is a 32-bit
   * integer with the value 0x484F4C4F or 0x4F4C4F48 depending on the
   * endianness, which corresponds to the ASCII string "HOLO".
   */
  uint32_t magic_number;

  /**
   * @brief Version number of the file format.
   *
   * The version number is used to identify the version of the file format.
   */
  uint16_t version;

  /**
   * @brief Number of bits per pixel.
   *
   * The number of bits per pixel is used to determine the black and white
   * depth of the frames.
   */
  uint16_t bits_per_pixel;

  /**
   * @brief Width of the frames in pixels.
   */
  uint32_t frame_width;

  /**
   * @brief Height of the frames in pixels.
   */
  uint32_t frame_height;

  /**
   * @brief Number of frames in the file.
   */
  uint32_t frame_count;

  /**
   * @brief Total size of the data section in bytes.
   */
  uint64_t total_data_size;

  /**
   * @brief Endianness of the file.
   *
   * The endianness of the file is used to determine the byte order of the
   * data section.
   */
  uint8_t endianness;

  /**
   * @brief Reserved padding.
   */
  char padding[35];
};
#pragma pack(pop)

// We need to ensure the header is exactly 64 bytes.
static_assert(sizeof(HolofileHeader) == 64, "HolofileHeader must be 64 bytes");

class HolofileReader {
public:
  /**
   * Opens a HolofileReader for the specified filename.
   *
   * @param filename The name of the file to open.
   * @return A `tl::expected` containing a `HolofileReader` if the file was
   * successfully opened, or a `std::error_code` if an error occurred. The
   * possible error codes are:
   * - `dh::HolofileError::IncompleteHeader`: The end of the file was
   * reached unexpectedly while reading the header.
   * - `dh::HolofileError::InvalidMagicNumber`: The magic number in the
   * header is invalid.
   * - `dh::HolofileError::InvalidVersion`: The version number in the
   * header is invalid.
   * - `dh::HolofileError::InvalidFrameSize`: The frame size in the header
   * is not a multiple of 8 bits.
   * - Underlying file I/O errors accessed through `std::generic_category`.
   */
  static tl::expected<HolofileReader, std::error_code>
  open(const std::string &filename);

  /**
   * Gets the header of the Holofile.
   *
   * @return A reference to the header of the Holofile.
   */
  const HolofileHeader &header() const;

  /**
   * Seeks to the specified frame index in the Holofile.
   *
   * @param frame_index The index of the frame to seek to.
   * @return A `tl::expected` indicating success or failure. If an error
   * occurs, a `std::error_code` is returned. The possible error codes are:
   * - Underlying file I/O errors accessed through `std::generic_category`.
   */
  tl::expected<void, std::error_code> seek(size_t frame_index);

  /**
   * Gets the current frame index.
   *
   * @return The current frame index.
   */
  size_t tell() const;

  /**
   * Reads the specified number of frames from the Holofile.
   *
   * @param data A pointer to the buffer to read the frames into.
   * @param count The number of frames to read.
   * @return A `tl::expected` indicating success or failure. If an error
   * occurs, a `std::error_code` is returned. The possible error codes are:
   * - `dh::HolofileError::EndOfFile`: The end of the file was reached
   * unexpectedly while reading the frames.
   * - Underlying file I/O errors accessed through `std::generic_category`.
   */
  tl::expected<void, std::error_code> read_frames(uint8_t *data, size_t count);

private:
  /**
   * @brief Deleter for FILE pointers.
   *
   * This class is used to delete FILE pointers when they go out of scope.
   */
  struct FileCloser {
    void operator()(FILE *file) const { fclose(file); }
  };

  /**
   * @brief Constructor for HolofileReader.
   *
   * This constructor is private and should only be called by the `open`
   * method.
   *
   * @param file The file pointer to the Holofile.
   * @param header The header of the Holofile.
   */
  HolofileReader(std::unique_ptr<FILE, FileCloser> file,
                 const HolofileHeader &header);

  /**
   * @brief The file pointer to the Holofile.
   */
  std::unique_ptr<FILE, FileCloser> file_;

  /**
   * @brief The header of the Holofile.
   */
  HolofileHeader header_;

  /**
   * @brief The current frame index.
   */
  size_t frame_index_;
};
}; // namespace dh