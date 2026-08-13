#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RAPIDLLM_API_VERSION 1

typedef enum RapidStatus {
    RAPID_OK = 0,
    RAPID_ERR_IO,
    RAPID_ERR_FORMAT,
    RAPID_ERR_NOMEM,
    RAPID_ERR_UNSUPPORTED,
    RAPID_ERR_DEVICE,
    RAPID_ERR_RANGE,
    RAPID_ERR_INTERNAL
} RapidStatus;

typedef struct RapidError {
    RapidStatus code;
    char message[256];
} RapidError;

typedef struct RapidConfig {
    const char* model_path;
    const char* device;
    int threads;
    int ctx;
    int kv_i8;
    int fuse;
    int mmap;
    int hugepage;
    int language_only;
    int repack_int4;
    int max_layers;
    unsigned long long max_ram_bytes;
} RapidConfig;

typedef struct RapidSessionConfig {
    int enable_thinking;
    int preserve_thinking;
    int max_new_tokens;
    int spec;    /* 0=off 1=ngram 2=mtp 3=auto 4=draft-model */
    int spec_n;  /* draft tokens per step */
} RapidSessionConfig;

typedef struct RapidSpecStats {
    int proposed;
    int accepted;
    int steps;
} RapidSpecStats;

typedef struct RapidSampleParams {
    float temperature, top_p, min_p;
    float presence_penalty, repetition_penalty;
    int top_k;
    int greedy;
} RapidSampleParams;

typedef struct RapidLogitsView {
    const float* data;
    int vocab;
} RapidLogitsView;

typedef struct RapidLLM RapidLLM;
typedef struct RapidSession RapidSession;

void          rapidllm_set_fuse(RapidLLM*, int fuse);

RapidLLM*     rapidllm_load(const RapidConfig*, RapidError*);
void          rapidllm_free(RapidLLM*);
int           rapidllm_vocab(const RapidLLM*);
int           rapidllm_encode(RapidLLM*, const char* utf8, int32_t* ids, int cap, RapidError*);
int           rapidllm_decode_ids(RapidLLM*, const int32_t*, int n, char* out, int cap, RapidError*);
int           rapidllm_generate(RapidSession*, const int32_t* ids, int n,
                                const RapidSampleParams*, int32_t* out, int cap, RapidError*);
/* Continuous batch: n_seq copies of the same prompt, one shared weight pass per step. */
int           rapidllm_generate_batch(RapidSession*, const int32_t* ids, int n, int n_seq,
                                      const RapidSampleParams*, int32_t* out, int cap_each, int* out_n,
                                      RapidError*);
RapidSession* rapidllm_session_new(RapidLLM*, const RapidSessionConfig*, RapidError*);
void          rapidllm_session_set_max_new(RapidSession*, int max_new_tokens);
int           rapidllm_session_set_draft(RapidSession* target, RapidSession* draft, RapidError*);
int           rapidllm_prefill(RapidSession*, const int32_t*, int n, RapidError*);
int           rapidllm_decode(RapidSession*, int32_t token, RapidLogitsView*, RapidError*);
int           rapidllm_sample(RapidSession*, const RapidSampleParams*, int32_t* token, RapidError*);
void          rapidllm_session_free(RapidSession*);
void          rapidllm_spec_stats(RapidSession*, RapidSpecStats*);
void          rapidllm_bench_stats(RapidSession*, double* prefill_s, double* decode_s, int* n_decode);
int           rapidllm_uses_cuda(const RapidSession*);
int           rapidllm_api_version(void);
const char*   rapidllm_version_string(void);

#ifdef __cplusplus
}
#endif
