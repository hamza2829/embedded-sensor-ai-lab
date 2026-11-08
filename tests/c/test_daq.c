#include "daq.h"

#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Tiny in-tree test harness. No external framework or installation is needed. */
#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); return 1; \
} } while (0)

static void put_u32(uint8_t *bytes, uint32_t value) {
    for (unsigned index = 0; index < 4; ++index) bytes[index] = (uint8_t)(value >> (8u * index));
}

static void put_float(uint8_t *bytes, float value) {
    uint32_t bits;
    memcpy(&bits, &value, 4);
    put_u32(bytes, bits);
}

static void make_frame(uint8_t *bytes, uint32_t sequence, uint64_t time,
                       float temperature, float vibration, float current) {
    memcpy(bytes, "SENS", 4);
    bytes[4] = 1; bytes[5] = 0; bytes[6] = 24; bytes[7] = 0;
    put_u32(bytes + 8, sequence);
    put_u32(bytes + 12, (uint32_t)time);
    put_u32(bytes + 16, (uint32_t)(time >> 32));
    put_float(bytes + 20, temperature);
    put_float(bytes + 24, vibration);
    put_float(bytes + 28, current);
    put_u32(bytes + 32, daq_crc32(bytes, 32));
}

static void emit(FILE *file, uint32_t sequence, uint64_t time) {
    uint8_t frame[DAQ_FRAME_SIZE];
    make_frame(frame, sequence, time, 25.5f, -0.25f, 2.0f);
    if (fwrite(frame, 1, sizeof(frame), file) != sizeof(frame)) abort();
}

static int test_crc(void) {
    CHECK(daq_crc32((const uint8_t *)"123456789", 9) == UINT32_C(0xcbf43926));
    CHECK(daq_crc32((const uint8_t *)"", 0) == 0);
    return 0;
}

static int test_decode(void) {
    uint8_t frame[DAQ_FRAME_SIZE];
    daq_sample sample;
    make_frame(frame, UINT32_C(0x01020304), UINT64_C(0x1020304050607080), 25.5f, -0.25f, 2.0f);
    CHECK(daq_decode(frame, &sample) == DAQ_OK);
    CHECK(sample.sequence == UINT32_C(0x01020304));
    CHECK(sample.source_us == UINT64_C(0x1020304050607080));
    CHECK(sample.temperature == 25.5f && sample.vibration == -0.25f && sample.current == 2.0f);
    return 0;
}

static int test_crc_reject(void) {
    uint8_t frame[DAQ_FRAME_SIZE];
    daq_sample sample;
    make_frame(frame, 0, 0, 25, 0, 2);
    frame[20] ^= 1;
    CHECK(daq_decode(frame, &sample) == DAQ_BAD_CRC);
    return 0;
}

static int test_header_reject(void) {
    uint8_t frame[DAQ_FRAME_SIZE];
    daq_sample sample;
    const unsigned offsets[] = {0, 4, 5, 6, 7};
    for (unsigned index = 0; index < sizeof(offsets) / sizeof(offsets[0]); ++index) {
        make_frame(frame, 0, 0, 25, 0, 2);
        frame[offsets[index]] ^= 2;
        CHECK(daq_decode(frame, &sample) == DAQ_BAD_HEADER);
    }
    return 0;
}

static int test_values_reject(void) {
    uint8_t frame[DAQ_FRAME_SIZE];
    daq_sample sample;
    const float bad[][3] = {{NAN, 0, 2}, {25, INFINITY, 2}, {25, 0, NAN},
                            {-51, 0, 2}, {151, 0, 2}, {25, 101, 2},
                            {25, 0, -51}, {25, 0, 101}};
    for (unsigned index = 0; index < sizeof(bad) / sizeof(bad[0]); ++index) {
        make_frame(frame, 0, 0, bad[index][0], bad[index][1], bad[index][2]);
        CHECK(daq_decode(frame, &sample) == DAQ_BAD_VALUES);
    }
    return 0;
}

static int test_log(void) {
    FILE *input = tmpfile(), *output = tmpfile(), *metrics = tmpfile();
    CHECK(input != NULL && output != NULL && metrics != NULL);
    emit(input, 0, 0); emit(input, 1, 5000); rewind(input);
    daq_stats stats;
    CHECK(daq_process(input, output, &stats) == 0);
    CHECK(stats.accepted == 2 && stats.frames == 2 && stats.sequence_gaps == 0);
    rewind(output);
    char line[256];
    CHECK(fgets(line, sizeof(line), output) != NULL && strstr(line, "acquired_ns") != NULL);
    CHECK(fgets(line, sizeof(line), output) != NULL);
    uint32_t sequence;
    uint64_t source, acquired;
    double temperature, vibration, current;
    CHECK(sscanf(line, "%" SCNu32 ",%" SCNu64 ",%" SCNu64 ",%lf,%lf,%lf",
                 &sequence, &source, &acquired, &temperature, &vibration, &current) == 6);
    CHECK(sequence == 0 && source == 0 && acquired > 0 && temperature == 25.5);
    CHECK(daq_write_stats(metrics, &stats) == 0);
    rewind(metrics);
    CHECK(fgets(line, sizeof(line), metrics) != NULL && strstr(line, "\"accepted\":2") != NULL);
    fclose(input); fclose(output); fclose(metrics);
    return 0;
}

static int test_gaps(void) {
    FILE *input = tmpfile(), *output = tmpfile();
    CHECK(input != NULL && output != NULL);
    emit(input, 2, 10000); emit(input, 5, 25000); rewind(input);
    daq_stats stats;
    CHECK(daq_process(input, output, &stats) == 0);
    CHECK(stats.accepted == 2 && stats.sequence_gaps == 4);
    fclose(input); fclose(output);
    return 0;
}

static int test_order(void) {
    FILE *input = tmpfile(), *output = tmpfile();
    CHECK(input != NULL && output != NULL);
    emit(input, 0, 0); emit(input, 0, 5000); emit(input, 1, 0); emit(input, 1, 5000); rewind(input);
    daq_stats stats;
    CHECK(daq_process(input, output, &stats) == 0);
    CHECK(stats.accepted == 2 && stats.out_of_order == 2);
    fclose(input); fclose(output);
    return 0;
}

static int test_truncated(void) {
    FILE *input = tmpfile(), *output = tmpfile();
    CHECK(input != NULL && output != NULL);
    emit(input, 0, 0);
    CHECK(fwrite("abc", 1, 3, input) == 3);
    rewind(input);
    daq_stats stats;
    CHECK(daq_process(input, output, &stats) == 0);
    CHECK(stats.accepted == 1 && stats.truncated_bytes == 3);
    fclose(input); fclose(output);
    return 0;
}

static int test_empty(void) {
    FILE *input = tmpfile(), *output = tmpfile();
    CHECK(input != NULL && output != NULL);
    daq_stats stats;
    CHECK(daq_process(input, output, &stats) == 0);
    CHECK(stats.frames == 0 && stats.accepted == 0);
    fclose(input); fclose(output);
    return 0;
}

static int test_rejects(void) {
    FILE *input = tmpfile(), *output = tmpfile();
    CHECK(input != NULL && output != NULL);
    uint8_t frame[DAQ_FRAME_SIZE];
    for (unsigned index = 0; index < 3; ++index) {
        make_frame(frame, index, index * 5000, index == 2 ? NAN : 25, 0, 2);
        if (index == 0) frame[32] ^= 1;
        if (index == 1) frame[0] ^= 1;
        CHECK(fwrite(frame, 1, sizeof(frame), input) == sizeof(frame));
    }
    rewind(input);
    daq_stats stats;
    CHECK(daq_process(input, output, &stats) == 0);
    CHECK(stats.frames == 3 && stats.accepted == 0);
    CHECK(stats.bad_crc == 1 && stats.bad_header == 1 && stats.bad_values == 1);
    fclose(input); fclose(output);
    return 0;
}

int main(int argc, char **argv) {
    const struct { const char *name; int (*run)(void); } cases[] = {
        {"crc", test_crc}, {"decode", test_decode}, {"crc_reject", test_crc_reject},
        {"header_reject", test_header_reject}, {"values_reject", test_values_reject},
        {"log", test_log}, {"gaps", test_gaps}, {"order", test_order},
        {"truncated", test_truncated}, {"empty", test_empty}, {"rejects", test_rejects}
    };
    if (argc != 2) return 2;
    for (unsigned index = 0; index < sizeof(cases) / sizeof(cases[0]); ++index) {
        if (strcmp(argv[1], cases[index].name) == 0) return cases[index].run();
    }
    return 2;
}
