#include "holofile.hh"
#include <bit>
#include <iostream>

namespace dh {
const char *holofile_error_category::name() const noexcept {
  return "holofile";
}

std::string holofile_error_category::message(int ev) const {
  switch (static_cast<HolofileError>(ev)) {
  case HolofileError::EndOfFile:
    return "End of file";
  case HolofileError::IncompleteHeader:
    return "Incomplete header";
  case HolofileError::InvalidMagicNumber:
    return "Invalid magic number";
  case HolofileError::InvalidVersion:
    return "Invalid version";
  case HolofileError::InvalidFrameSize:
    return "Invalid frame size";
  default:
    return "Unknown error";
  }
}

const std::error_category &holofile_category() {
  static holofile_error_category instance;
  return instance;
}

std::error_code make_error_code(HolofileError e) {
  return {static_cast<int>(e), holofile_category()};
}

tl::expected<HolofileReader, std::error_code>
HolofileReader::open(const std::string &filename) {
  // Open the file for reading.
  std::unique_ptr<FILE, FileCloser> file(fopen(filename.c_str(), "rb"));
  if (!file)
    return tl::unexpected(std::error_code(errno, std::generic_category()));

  // Read the header.
  HolofileHeader header;
  size_t success = fread(&header, sizeof(HolofileHeader), 1, file.get());

  // TODO: Check if one should use clearerr() here.
  if (ferror(file.get()) != 0)
    return tl::unexpected(std::error_code(errno, std::generic_category()));

  // TODO: Check if one should use clearerr() here.
  if (feof(file.get()) != 0)
    return tl::unexpected(make_error_code(HolofileError::IncompleteHeader));

  if (!success) {
    std::cerr << "[holofile] Unrecoverable error: fread() failed to read the "
                 "header.\n";
    std::exit(EXIT_FAILURE);
  }

  // Check the header.
  uint32_t magic_number =
      std::endian::native == std::endian::little ? 0x484F4C4F : 0x4F4C4F48;

  if (header.magic_number != magic_number)
    return tl::unexpected(make_error_code(HolofileError::InvalidMagicNumber));

  if (header.version != 2)
    return tl::unexpected(make_error_code(HolofileError::InvalidVersion));

  size_t pixels_per_frame = header.frame_width * header.frame_height;
  size_t bits_per_frame = pixels_per_frame * header.bits_per_pixel;
  if (bits_per_frame % 8 != 0)
    return tl::unexpected(make_error_code(HolofileError::InvalidFrameSize));

  return HolofileReader(std::move(file), header);
}

const HolofileHeader &HolofileReader::header() const { return header_; }

tl::expected<void, std::error_code> HolofileReader::seek(size_t frame_index) {
  size_t pixels_per_frame = header_.frame_width * header_.frame_height;
  size_t bits_per_frame = pixels_per_frame * header_.bits_per_pixel;
  size_t bytes_per_frame = bits_per_frame / 8;

  size_t offset = sizeof(HolofileHeader) + frame_index * bytes_per_frame;
  if (fseek(file_.get(), offset, SEEK_SET) != 0)
    return tl::unexpected(std::error_code(errno, std::generic_category()));

  frame_index_ = frame_index;
  return {};
}

size_t HolofileReader::tell() const { return frame_index_; }

tl::expected<void, std::error_code> HolofileReader::read_frames(uint8_t *data,
                                                                size_t count) {
  size_t pixels_per_frame = header_.frame_width * header_.frame_height;
  size_t bits_per_frame = pixels_per_frame * header_.bits_per_pixel;
  size_t bytes_per_frame = bits_per_frame / 8;
  size_t frames_read = fread(data, bytes_per_frame, count, file_.get());

  frame_index_ += frames_read;

  // TODO: Check if one should use clearerr() here.
  if (ferror(file_.get()) != 0)
    return tl::unexpected(std::error_code(errno, std::generic_category()));

  // TODO: Check if one should use clearerr() here.
  if (feof(file_.get()) != 0)
    return tl::unexpected(make_error_code(HolofileError::EndOfFile));

  if (frames_read != count) {
    std::cerr << "[holofile] Unrecoverable error: fread() failed to read the "
                 "requested number of frames.\n";
    std::exit(EXIT_FAILURE);
  }

  return {};
}

HolofileReader::HolofileReader(std::unique_ptr<FILE, FileCloser> file,
                               const HolofileHeader &header)
    : file_(std::move(file)), header_(header), frame_index_(0) {}

} // namespace dh