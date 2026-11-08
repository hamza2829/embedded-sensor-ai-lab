#include "daq.h"

#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <string.h>
#include <time.h>

_Static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24,
               "wire format requires IEEE 754 binary32 floats");

static uint32_t read_u32(const uint8_t *data) {
    return (uint32_t)data[0] | ((uint32_t)data[1] << 8) |
           ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
}

static uint64_t read_u64(const uint8_t *data) {
    return (uint64_t)read_u32(data) | ((uint64_t)read_u32(data + 4) << 32);
}

static float read_float(const uint8_t *data) {
    const uint32_t bits = read_u32(data);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

uint32_t daq_crc32(const uint8_t *data, size_t length) {
    uint32_t crc = UINT32_C(0xffffffff);
    for (size_t index = 0; index < length; ++index) {
        crc ^= data[index];
        for (unsigned bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^ ((crc & 1u) ? UINT32_C(0xedb88320) : 0u);
        }
    }
    return crc ^ UINT32_C(0xffffffff);
}

daq_status daq_decode(const uint8_t frame[DAQ_FRAME_SIZE], daq_sample *sample) {
    if (memcmp(frame, "SENS", 4) != 0 || frame[4] != 1 || frame[5] != 0 ||
        frame[6] != 24 || frame[7] != 0) {
        return DAQ_BAD_HEADER;
    }
    if (daq_crc32(frame, 32) != read_u32(frame + 32)) {
        return DAQ_BAD_CRC;
    }
    sample->sequence = read_u32(frame + 8);
    sample->source_us = read_u64(frame + 12);
    sample->temperature = read_float(frame + 20);
    sample->vibration = read_float(frame + 24);
    sample->current = read_float(frame + 28);
    if (!isfinite(sample->temperature) || !isfinite(sample->vibration) ||
        !isfinite(sample->current) || sample->temperature < -50 ||
        sample->temperature > 150 || fabsf(sample->vibration) > 100 ||
        sample->current < -50 || sample->current > 100) {
        return DAQ_BAD_VALUES;
    }
    return DAQ_OK;
}

int daq_process(FILE *input, FILE *output, daq_stats *stats) {
    uint8_t frame[DAQ_FRAME_SIZE];
    uint32_t previous_sequence = 0;
    uint64_t previous_time = 0;
    int have_previous = 0;
    memset(stats, 0, sizeof(*stats));
    if (fprintf(output, "sequence,source_us,acquired_ns,temperature,vibration,current\n") < 0) {
        return -1;
    }
    size_t length;
    while ((length = fread(frame, 1, sizeof(frame), input)) == sizeof(frame)) {
        daq_sample sample;
        stats->frames++;
        const daq_status status = daq_decode(frame, &sample);
        if (status != DAQ_OK) {
            if (status == DAQ_BAD_CRC) stats->bad_crc++;
            else if (status == DAQ_BAD_HEADER) stats->bad_header++;
            else stats->bad_values++;
            continue;
        }
        if (have_previous && (sample.sequence <= previous_sequence ||
                              sample.source_us <= previous_time)) {
            stats->out_of_order++;
            continue;
        }
        if (have_previous) {
            stats->sequence_gaps += (uint64_t)sample.sequence - previous_sequence - 1u;
        } else {
            stats->sequence_gaps += sample.sequence;
        }
        struct timespec now;
        if (timespec_get(&now, TIME_UTC) != TIME_UTC) return -1;
        const uint64_t acquired_ns = (uint64_t)now.tv_sec * UINT64_C(1000000000) +
                                     (uint64_t)now.tv_nsec;
        if (fprintf(output, "%" PRIu32 ",%" PRIu64 ",%" PRIu64 ",%.9g,%.9g,%.9g\n",
                    sample.sequence, sample.source_us, acquired_ns,
                    (double)sample.temperature, (double)sample.vibration,
                    (double)sample.current) < 0) return -1;
        stats->accepted++;
        previous_sequence = sample.sequence;
        previous_time = sample.source_us;
        have_previous = 1;
    }
    stats->truncated_bytes = length;
    return ferror(input) || fflush(output) != 0 ? -1 : 0;
}

int daq_write_stats(FILE *output, const daq_stats *stats) {
    const int result = fprintf(output,
        "{\"frames\":%" PRIu64 ",\"accepted\":%" PRIu64
        ",\"bad_header\":%" PRIu64 ",\"bad_crc\":%" PRIu64
        ",\"bad_values\":%" PRIu64 ",\"sequence_gaps\":%" PRIu64
        ",\"out_of_order\":%" PRIu64 ",\"truncated_bytes\":%" PRIu64 "}\n",
        stats->frames, stats->accepted, stats->bad_header, stats->bad_crc,
        stats->bad_values, stats->sequence_gaps, stats->out_of_order, stats->truncated_bytes);
    return result < 0 || fflush(output) != 0 ? -1 : 0;
}
