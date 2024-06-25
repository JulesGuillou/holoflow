#include <gtest/gtest.h>
#include <holofile.hh>

namespace dh {
class HolofileReaderHeaderTests
    : public ::testing::TestWithParam<
          std::tuple<HolofileHeader, std::error_code>> {};

TEST_P(HolofileReaderHeaderTests, Header) {
  // Test parameters.
  const auto [header, expected_error] = GetParam();

  // Create a temporary file.
  FILE *file = fopen("test.holo", "wb");
  ASSERT_NE(file, nullptr);

  // Write the header to the file.
  size_t success = fwrite(&header, sizeof(HolofileHeader), 1, file);
  ASSERT_EQ(success, 1);

  // Close the file.
  ASSERT_EQ(fclose(file), 0);

  // Create a HolofileReader.
  auto reader = HolofileReader::open("test.holo");
  if (expected_error) {
    ASSERT_FALSE(reader);
    ASSERT_EQ(reader.error(), expected_error);
  } else {
    if (!reader) {
      std::cerr << "Error: " << reader.error().message() << std::endl;
    }
    ASSERT_TRUE(reader);
    ASSERT_EQ(reader->header().magic_number, header.magic_number);
    ASSERT_EQ(reader->header().version, header.version);
    ASSERT_EQ(reader->header().bits_per_pixel, header.bits_per_pixel);
    ASSERT_EQ(reader->header().frame_width, header.frame_width);
    ASSERT_EQ(reader->header().frame_height, header.frame_height);
    ASSERT_EQ(reader->header().frame_count, header.frame_count);
    ASSERT_EQ(reader->header().total_data_size, header.total_data_size);
    ASSERT_EQ(reader->header().endianness, header.endianness);
    ASSERT_EQ(memcmp(reader->header().padding, header.padding, 35), 0);
  }

  // Remove the temporary file.
  remove("test.holo");
}

INSTANTIATE_TEST_SUITE_P(
    HolofileReaderHeaderTests, HolofileReaderHeaderTests,
    ::testing::Values(
        // Valid header.
        std::make_tuple(
            HolofileHeader{.magic_number = 0x484F4C4F,
                           .version = 2,
                           .bits_per_pixel = 8,
                           .frame_width = 15,
                           .frame_height = 15,
                           .frame_count = 10,
                           .total_data_size = 15 * 15 * 10,
                           .endianness = (uint8_t)std::endian::little,
                           .padding = {0}},
            std::error_code{}),
        // Invalid magic number.
        std::make_tuple(
            HolofileHeader{.magic_number = 0xDEADBEEF,
                           .version = 2,
                           .bits_per_pixel = 8,
                           .frame_width = 15,
                           .frame_height = 15,
                           .frame_count = 10,
                           .total_data_size = 15 * 15 * 10,
                           .endianness = (uint8_t)std::endian::little,
                           .padding = {0}},
            make_error_code(HolofileError::InvalidMagicNumber)),
        // Invalid version.
        std::make_tuple(
            HolofileHeader{.magic_number = 0x484F4C4F,
                           .version = 1,
                           .bits_per_pixel = 8,
                           .frame_width = 15,
                           .frame_height = 15,
                           .frame_count = 10,
                           .total_data_size = 15 * 15 * 10,
                           .endianness = (uint8_t)std::endian::little,
                           .padding = {0}},
            make_error_code(HolofileError::InvalidVersion)),
        // Invalid frame size.
        std::make_tuple(
            HolofileHeader{.magic_number = 0x484F4C4F,
                           .version = 2,
                           .bits_per_pixel = 4,
                           .frame_width = 15,
                           .frame_height = 15,
                           .frame_count = 10,
                           .total_data_size = 15 * 15 * 10,
                           .endianness = (uint8_t)std::endian::little,
                           .padding = {0}},
            make_error_code(HolofileError::InvalidFrameSize))));

TEST(HolofileReaderHeaderTest, IncompleteHeader) {
  // Create a temporary file.
  FILE *file = fopen("test.holo", "wb");
  ASSERT_NE(file, nullptr);

  // Write an incomplete header to the file.
  char header[sizeof(HolofileHeader) - 1];
  size_t bytes_written = fwrite(header, sizeof(char), sizeof(header), file);
  ASSERT_EQ(bytes_written, sizeof(header));

  // Close the file.
  ASSERT_EQ(fclose(file), 0);

  // Create a HolofileReader.
  auto reader = HolofileReader::open("test.holo");
  ASSERT_FALSE(reader);
  ASSERT_EQ(reader.error(), make_error_code(HolofileError::IncompleteHeader));

  // Remove the temporary file.
  remove("test.holo");
}

class HolofileReaderContentTests : public ::testing::Test {
protected:
  void SetUp() override {
    // Create a temporary file.
    FILE *file = fopen("test.holo", "wb");
    ASSERT_NE(file, nullptr);

    // Write a valid header to the file.
    HolofileHeader header = {.magic_number = 0x484F4C4F,
                             .version = 2,
                             .bits_per_pixel = 8,
                             .frame_width = 15,
                             .frame_height = 15,
                             .frame_count = 10,
                             .total_data_size = 15 * 15 * 10,
                             .endianness = (uint8_t)std::endian::little,
                             .padding = {0}};
    size_t success = fwrite(&header, sizeof(HolofileHeader), 1, file);
    ASSERT_EQ(success, 1);

    // Write the data to the file.
    uint8_t counter = 0;
    uint8_t data[15 * 15 * 10];
    for (size_t i = 0; i < 10 * 15 * 15; i++)
      data[i] = counter++;

    size_t bytes_written = fwrite(data, sizeof(uint8_t), sizeof(data), file);
    ASSERT_EQ(bytes_written, sizeof(data));

    // Close the file.
    ASSERT_EQ(fclose(file), 0);
  }

  void TearDown() override { remove("test.holo"); }
};

TEST_F(HolofileReaderContentTests, ReadAllFramesAtOnce) {
  // Create a HolofileReader.
  auto reader = HolofileReader::open("test.holo");
  ASSERT_TRUE(reader);

  // Read all frames.
  uint8_t counter = 0;
  uint8_t data[15 * 15 * 10];
  auto result = reader->read_frames(data, 10);
  ASSERT_TRUE(result);
  for (size_t i = 0; i < 10 * 15 * 15; i++)
    ASSERT_EQ(data[i], counter++);
}

TEST_F(HolofileReaderContentTests, ReadFramesOneByOne) {
  // Create a HolofileReader.
  auto reader = HolofileReader::open("test.holo");
  ASSERT_TRUE(reader);

  // Read frames one by one.
  uint8_t counter = 0;
  uint8_t data[15 * 15];
  for (size_t i = 0; i < 10; i++) {
    auto result = reader->read_frames(data, 1);
    ASSERT_TRUE(result);
    for (size_t j = 0; j < 15 * 15; j++)
      ASSERT_EQ(data[j], counter++);
  }
}

TEST_F(HolofileReaderContentTests, ReadFramesEndOfFile) {
  // Create a HolofileReader.
  auto reader = HolofileReader::open("test.holo");
  ASSERT_TRUE(reader);

  // Read all frames.
  uint8_t data[15 * 15 * 10];
  auto result = reader->read_frames(data, 11);
  ASSERT_FALSE(result);
  ASSERT_EQ(result.error(), make_error_code(HolofileError::EndOfFile));
}

TEST_F(HolofileReaderContentTests, SeekBeginning) {
  // Create a HolofileReader.
  auto reader = HolofileReader::open("test.holo");
  ASSERT_TRUE(reader);

  // Read the first frame.
  uint8_t data[15 * 15];
  auto result = reader->read_frames(data, 1);
  ASSERT_TRUE(result);

  // Seek to the beginning.
  result = reader->seek(0);
  ASSERT_TRUE(result);

  // Read the first frame again.
  uint8_t counter = 0;
  result = reader->read_frames(data, 1);
  ASSERT_TRUE(result);
  for (size_t i = 0; i < 15 * 15; i++)
    ASSERT_EQ(data[i], counter++);
}

TEST_F(HolofileReaderContentTests, SeekMiddle) {
  // Create a HolofileReader.
  auto reader = HolofileReader::open("test.holo");
  ASSERT_TRUE(reader);

  // Read the first frame.
  uint8_t data[15 * 15];
  auto result = reader->read_frames(data, 1);
  ASSERT_TRUE(result);

  // Seek to the middle.
  result = reader->seek(5);
  ASSERT_TRUE(result);

  // Read the sixth frame.
  uint8_t counter = static_cast<uint8_t>(5 * 15 * 15);
  result = reader->read_frames(data, 1);
  ASSERT_TRUE(result);
  for (size_t i = 0; i < 15 * 15; i++)
    ASSERT_EQ(data[i], counter++);
}

TEST_F(HolofileReaderContentTests, SeekEnd) {
  // Create a HolofileReader.
  auto reader = HolofileReader::open("test.holo");
  ASSERT_TRUE(reader);

  // Read the first frame.
  uint8_t data[15 * 15];
  auto result = reader->read_frames(data, 1);
  ASSERT_TRUE(result);

  // Seek to the end.
  result = reader->seek(10);
  ASSERT_TRUE(result);

  // Read the last frame. This should fail.
  result = reader->read_frames(data, 1);
  ASSERT_FALSE(result);
  ASSERT_EQ(result.error(), make_error_code(HolofileError::EndOfFile));
}

TEST_F(HolofileReaderContentTests, TellAfterReadFrames) {
  // Create a HolofileReader.
  auto reader = HolofileReader::open("test.holo");
  ASSERT_TRUE(reader);

  // Check the initial frame index.
  ASSERT_EQ(reader->tell(), 0);

  for (size_t i = 0; i < 10; i++) {
    // Read the first frame.
    uint8_t data[15 * 15];
    auto result = reader->read_frames(data, 1);
    ASSERT_TRUE(result);

    // Check the current frame index.
    ASSERT_EQ(reader->tell(), i + 1);
  }
}

TEST_F(HolofileReaderContentTests, TellAfterSeek) {
  // Create a HolofileReader.
  auto reader = HolofileReader::open("test.holo");
  ASSERT_TRUE(reader);

  // Check the initial frame index.
  ASSERT_EQ(reader->tell(), 0);

  for (size_t i = 0; i < 10; i++) {
    // Seek to the middle.
    auto result = reader->seek(i);
    ASSERT_TRUE(result);

    // Check the current frame index.
    ASSERT_EQ(reader->tell(), i);
  }
}

TEST_F(HolofileReaderContentTests, SeekAfterEOF) {
  // Create a HolofileReader.
  auto reader = HolofileReader::open("test.holo");
  ASSERT_TRUE(reader);

  // Read until the end.
  uint8_t data[15 * 15];
  for (size_t i = 0; i < 10; i++) {
    auto result = reader->read_frames(data, 1);
    ASSERT_TRUE(result);
  }

  // Should be EOF now.
  auto result = reader->read_frames(data, 1);
  ASSERT_FALSE(result);
  ASSERT_EQ(result.error(), make_error_code(HolofileError::EndOfFile));

  // Seek to the beginning.
  result = reader->seek(0);
  ASSERT_TRUE(result);

  // Read the first frame.
  result = reader->read_frames(data, 1);
  ASSERT_TRUE(result);
}
}; // namespace dh
