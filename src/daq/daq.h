#ifndef SENSOR_DAQ_H
#define SENSOR_DAQ_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define DAQ_FRAME_SIZE 36u

typedef struct {
    uint32_t sequence;
    uint64_t source_us;
    float temperature;
    float vibration;
    float current;
} daq_sample;

typedef enum { DAQ_OK, DAQ_BAD_HEADER, DAQ_BAD_CRC, DAQ_BAD_VALUES } daq_status;

typedef struct {
    uint64_t frames;
    uint64_t accepted;
    uint64_t bad_header;
    uint64_t bad_crc;
    uint64_t bad_values;
    uint64_t sequence_gaps;
    uint64_t out_of_order;
    uint64_t truncated_bytes;
} daq_stats;

uint32_t daq_crc32(const uint8_t *data, size_t length);
daq_status daq_decode(const uint8_t frame[DAQ_FRAME_SIZE], daq_sample *sample);
int daq_process(FILE *input, FILE *output, daq_stats *stats);
int daq_write_stats(FILE *output, const daq_stats *stats);

#endif
