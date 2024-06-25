#include "holofile.hh"
#include <iostream>

int main() {
  // Open a holofile
  auto result = dh::HolofileReader::open("example.holo");
  if (!result) {
    std::cerr << "Error: " << result.error().message() << std::endl;
    return 1;
  }

  // Read frames
  auto reader = std::move(result.value());
  size_t bytes_per_frame = reader.header().frame_width *
                           reader.header().frame_height *
                           reader.header().bits_per_pixel / 8;

  std::vector<uint8_t> frame(bytes_per_frame);
  while (reader.read_frames(frame.data(), 1)) {
    // Process frame
  }
}