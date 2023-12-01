package dev.langchain4j.model.llama;

@SuppressWarnings("unused")
public class LlamaCppBackend {

    private LlamaCppBackend() { }

    public enum LlamaVocabType {
        SPM(0), // SentencePiece
        BPE(1); // Byte Pair Encoding
        public final int id;

        LlamaVocabType(int id) {
            this.id = id;
        }
    }

    public enum LlamaTokenType {
        UNDEFINED(0),
        NORMAL(1),
        UNKNOWN(2),
        CONTROL(3),
        UNDER_DEFINED(4),
        UNUSED(5),
        BYTE(6);
        public final int id;

        LlamaTokenType(int id) {
            this.id = id;
        }
    }

    // model file types
    public enum LlamaFileType {
        ALL_F32(0),
        MOSTLY_F16(1),                  // except 1d tensors
        MOSTLY_Q4_0(2),                 // except 1d tensors
        MOSTLY_Q4_1(3),                 // except 1d tensors
        MOSTLY_Q4_1_SOME_F16(4),        // tok_embeddings.weight and output.weight are F16
        // MOSTLY_Q4_2(5),              // support has been removed
        // MOSTLY_Q4_3(6),              // support has been removed
        MOSTLY_Q8_0(7),                 // except 1d tensors
        MOSTLY_Q5_0(8),                 // except 1d tensors
        MOSTLY_Q5_1(9),                 // except 1d tensors
        MOSTLY_Q2_K(10),                // except 1d tensors
        MOSTLY_Q3_K_S(11),              // except 1d tensors
        MOSTLY_Q3_K_M(12),              // except 1d tensors
        MOSTLY_Q3_K_L(13),              // except 1d tensors
        MOSTLY_Q4_K_S(14),              // except 1d tensors
        MOSTLY_Q4_K_M(15),              // except 1d tensors
        MOSTLY_Q5_K_S(16),              // except 1d tensors
        MOSTLY_Q5_K_M(17),              // except 1d tensors
        MOSTLY_Q6_K(18),                // except 1d tensors
        GUESSED(1024);                  // not specified in the model file
        public final int id;

        LlamaFileType(int id) {
            this.id = id;
        }
    }

    public enum LlamaRopeScalingType {
        SCALING_UNSPECIFIED(-1),
        SCALING_NONE(0),
        SCALING_LINEAR(1),
        SCALING_YARN(2),
        SCALING_MAX_VALUE(LlamaRopeScalingType.SCALING_YARN.id);
        public final int id;

        LlamaRopeScalingType(int id) {
            this.id = id;
        }
    }

    public static class LlamaTokenData {
        public final int id;        // token id
        public final float logit;   // log-odds of the token
        public final float p;       // probability of the token

        public LlamaTokenData(int id, float logit, float p) {
            this.id = id;
            this.logit = logit;
            this.p = p;
        }
    }

    public static class LlamaTokenDataArray {
        public final long data;         // llama_token_data*
        public final long size;         // size of the array
        public final boolean sorted;

        public LlamaTokenDataArray(long data, long size, boolean sorted) {
            this.data = data;
            this.size = size;
            this.sorted = sorted;
        }
    }

    // Input data for llama_decode
    // A llama_batch object can contain input about one or many sequences
    // The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
    //
    // - token  : the token ids of the input (used when embd is NULL)
    // - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
    // - pos    : the positions of the respective token in the sequence
    // - seq_id : the sequence to which the respective token belongs
    // - logits : if zero, the logits for the respective token will not be output
    //
    public static class LlamaBatch {
        public final int nTokens;
        public final long token;    // llama_token*
        public final long embd;     // float*
        public final long pos;      // llama_pos*
        public final long nSeqId;   // int32_t*
        public final long seq_id;   // llama_seq_id**
        public final long logits;   // int8_t*

        // NOTE: helpers for smooth API transition - can be deprecated in the future
        //       for future-proof code, use the above fields instead and ignore everything below
        //
        // pos[i] = all_pos_0 + i*all_pos_1
        //
        public final int allPos0;   // llama_pos
        public final int allPos1;   // llama_pos
        public final int allSeqId;  // llama_seq_id

        public LlamaBatch(int nTokens, long token, long embd, long pos, long nSeqId,
                          long seqId, long logits, int allPos0, int allPos1, int allSeqId) {
            this.nTokens = nTokens;
            this.token = token;
            this.embd = embd;
            this.pos = pos;
            this.nSeqId = nSeqId;
            seq_id = seqId;
            this.logits = logits;
            this.allPos0 = allPos0;
            this.allPos1 = allPos1;
            this.allSeqId = allSeqId;
        }
    }

    public static class LlamaModelParams {
        public final int nGpuLayers;                // number of layers to store in VRAM
        public final int mainGpu;                   // the GPU that is used for scratch and small tensors
        public final long tensorSplit;              // [float*] how to split layers across multiple GPUs (size: LLAMA_MAX_DEVICES)
        public final long progressCallback;         // [llama_progress_callback] called with a progress value between 0 and 1, pass NULL to disable
        public final long progressCallbackUserData; // [void*] context pointer passed to the progress callback
        public final boolean vocabOnly;             // only load the vocabulary, no weights
        public final boolean useMmap;               // use mmap if possible
        public final boolean useMlock;              // force system to keep model in RAM

        public LlamaModelParams(int nGpuLayers, int mainGpu, long tensorSplit, long progressCallback,
                                long progressCallbackUserData, boolean vocabOnly, boolean useMmap, boolean useMlock) {
            this.nGpuLayers = nGpuLayers;
            this.mainGpu = mainGpu;
            this.tensorSplit = tensorSplit;
            this.progressCallback = progressCallback;
            this.progressCallbackUserData = progressCallbackUserData;
            this.vocabOnly = vocabOnly;
            this.useMmap = useMmap;
            this.useMlock = useMlock;
        }
    }

    public static class LlamaContextParams {
        public final int seed;              // RNG seed, -1 for random
        public final int nCtx;              // text context, 0 = from model
        public final int nBatch;            // prompt processing maximum batch size
        public final int nThreads;          // number of threads to use for generation
        public final int nThreadsBatch;     // number of threads to use for batch processing
        public final int ropeScalingType;   // RoPE scaling type, from `enum llama_rope_scaling_type`

        // ref: https://github.com/ggerganov/llama.cpp/pull/2054
        public final float ropeFreqBase;    // RoPE base frequency, 0 = from model
        public final float ropeFreqScale;   // RoPE frequency scaling factor, 0 = from model
        public final float yarnExtFactor;   // YaRN extrapolation mix factor, negative = from model
        public final float yarnAttnFactor;  // YaRN magnitude scaling factor
        public final float yarnBetaFast;    // YaRN low correction dim
        public final float yarnBetaSlow;    // YaRN high correction dim
        public final int yarnOrigCtx;       // YaRN original context size

        // Keep the booleans together to avoid misalignment during copy-by-value.
        public final boolean mulMatQ;       // if true, use experimental mul_mat_q kernels (DEPRECATED - always true)
        public final boolean f16Kv;         // use fp16 for KV cache, fp32 otherwise
        public final boolean logitsAll;     // the llama_eval() call computes all logits, not just the last one
        public final boolean embedding;     // embedding mode only

        public LlamaContextParams(int seed, int nCtx, int nBatch, int nThreads, int nThreadsBatch, int ropeScalingType,
                                  float ropeFreqBase, float ropeFreqScale, float yarnExtFactor, float yarnAttnFactor,
                                  float yarnBetaFast, float yarnBetaSlow, int yarnOrigCtx, boolean mulMatQ, boolean f16Kv,
                                  boolean logitsAll, boolean embedding) {
            this.seed = seed;
            this.nCtx = nCtx;
            this.nBatch = nBatch;
            this.nThreads = nThreads;
            this.nThreadsBatch = nThreadsBatch;
            this.ropeScalingType = ropeScalingType;
            this.ropeFreqBase = ropeFreqBase;
            this.ropeFreqScale = ropeFreqScale;
            this.yarnExtFactor = yarnExtFactor;
            this.yarnAttnFactor = yarnAttnFactor;
            this.yarnBetaFast = yarnBetaFast;
            this.yarnBetaSlow = yarnBetaSlow;
            this.yarnOrigCtx = yarnOrigCtx;
            this.mulMatQ = mulMatQ;
            this.f16Kv = f16Kv;
            this.logitsAll = logitsAll;
            this.embedding = embedding;
        }
    }

    // model quantization parameters
    public static class LlamaModelQuantizeParams {
        public final int nThread;                 // number of threads to use for quantizing, if <=0 will use std::thread::hardware_concurrency()
        public final int fType;                   // [llama_ftype] quantize to this llama_ftype
        public final boolean allowRequantize;     // allow quantizing non-f32/f16 tensors
        public final boolean quantizeOutputTensor;// quantize output.weight
        public final boolean onlyCopy;            // only copy tensors - ftype, allow_requantize and quantize_output_tensor are ignored
        public final boolean pure;                // disable k-quant mixtures and quantize all tensors to the same type

        public LlamaModelQuantizeParams(int nThread, int fType, boolean allowRequantize,
                                        boolean quantizeOutputTensor, boolean onlyCopy, boolean pure) {
            this.nThread = nThread;
            this.fType = fType;
            this.allowRequantize = allowRequantize;
            this.quantizeOutputTensor = quantizeOutputTensor;
            this.onlyCopy = onlyCopy;
            this.pure = pure;
        }
    }

    // grammar element type
    enum LlamaGreType {
        END(0),
        ALT(1),
        RULE_REF(2),
        CHAR(3),
        CHAR_NOT(4),
        CHAR_RNG_UPPER(5),
        CHAR_ALT(6);
        public final int id;

        LlamaGreType(int id) {
            this.id = id;
        }
    }

    public static class LlamaGrammarElement {
        public final int type;   // [llama_gretype]
        public final int value;  // Unicode code point or rule ID

        public LlamaGrammarElement(int type, int value) {
            this.type = type;
            this.value = value;
        }
    }

    // performance timing information
    public static class LlamaTimings {
        public final double tStartMs;
        public final double tEndMs;
        public final double tLoadMs;
        public final double tSampleMs;
        public final double tPEvalMs;
        public final double tEvalMs;
        public final int nSample;
        public final int nPEval;
        public final int nEval;

        public LlamaTimings(double tStartMs, double tEndMs, double tLoadMs, double tSampleMs, double tPEvalMs,
                            double tEvalMs, int nSample, int nPEval, int nEval) {
            this.tStartMs = tStartMs;
            this.tEndMs = tEndMs;
            this.tLoadMs = tLoadMs;
            this.tSampleMs = tSampleMs;
            this.tPEvalMs = tPEvalMs;
            this.tEvalMs = tEvalMs;
            this.nSample = nSample;
            this.nPEval = nPEval;
            this.nEval = nEval;
        }
    }

    /**
     * @return struct llama_model_params
     */
    public native static LlamaModelParams llamaModelDefaultParams();

    /**
     * @return struct llama_context_params
     */
    public native static LlamaContextParams llamaContextDefaultParams();

    /**
     * @return struct llama_model_quantize_params
     */
    public native static LlamaModelQuantizeParams llamaModelQuantizeDefaultParams();

    /**
     * Initialize the llama + ggml backend
     * Call once at the start of the program
     * @param numa [bool] If numa is true, use NUMA optimizations
     */
    public native static void llamaBackendInit(boolean numa);

    /**
     * Call once at the end of the program - currently only used for MPI
     */
    public native static void llamaBackendFree();

    /**
     * @param pathModel [const char *] Path to the model file
     * @param params [const struct llama_model_params] Model parameters
     * @return struct llama_model *
     */
    public native static long llamaLoadModelFromFile(String pathModel, LlamaModelParams params);

    /**
     * @param model [const struct llama_model *] model
     */
    public native static void llamaFreeModel(long model);

    /**
     *
     * @param model [const struct llama_model *] model
     * @param params [struct llama_context_params] context parameters
     * @return struct llama_context *
     */
    public native static long llamaNewContextWithModel(long model, LlamaContextParams params);

    /**
     * Frees all allocated memory
     * @param ctx [struct llama_context *] context
     */
    public native static void llamaFree(long ctx);

    public native static long llamaTimeUs();

    public native static int llamaMaxDevices();

    public native static boolean llamaMmapSupported();

    public native static boolean llamaMlockSupported();

    /**
     * @param ctx [const struct llama_context *] context
     * @return [const struct llama_model *] model
     */
    public native static long llamaGetModel(long ctx);

    /**
     * @param ctx [const struct llama_context *] context
     * @return [int]
     */
    public native static int llamaNCtx(long ctx);

    /**
     *
     * @param model [const struct llama_model *] model
     * @return [int]
     */
    public native static int llamaVocabType(long model);

    /**
     * @param model [const struct llama_model *] model
     * @return [int]
     */
    public native static int llamaNVocab(long model);

    /**
     * @param model [const struct llama_model *] model
     * @return [int]
     */
    public native static int llamaNCtxTrain(long model);

    /**
     * @param model [const struct llama_model *]
     * @return [int]
     */
    public native static int llamaNEmbd(long model);

    /**
     * Get the model's RoPE frequency scaling factor
     * @param model [const struct llama_model *] model
     * @return [float]
     */
    public native static float llamaRopeFreqScaleTrain(long model);

    /**
     * Functions to access the model's GGUF metadata scalar values
     * - The functions return the length of the string on success, or -1 on failure
     * - The output string is always null-terminated and cleared on failure
     * - GGUF array values are not supported by these functions
     * @param model [const struct llama_model *] model
     * @param key [const char *]
     * @param buf [char *]
     * @param bufSize [size_t]
     * @return [int]
     */
    public native static int llamaModelMetaValStr(long model, String key, byte[] buf, long bufSize);

    /**
     * Get the number of metadata key/value pairs
     * @param model [const struct llama_model *]
     * @return [int]
     */
    public native static int llamaModelMetaCount(long model);

    /**
     * Get metadata key name by index
     * @param model [const struct llama_model *]
     * @param i [int] index
     * @param buf [char *]
     * @param bufSize [size_t]
     * @return [int]
     */
    public native static int llamaModelMetaKeyByIndex(long model, int i, byte[] buf, long bufSize);

    /**
     * Get metadata value as a string by index
     * @param model [const struct llama_model *]
     * @param i [int] index
     * @param buf [char *]
     * @param bufSize [size_t]
     * @return [int]
     */
    public native static int llamaModelMetaValStrByIndex(long model, int i, byte[] buf, long bufSize);

    /**
     * Get a string describing the model type
     * @param model [const struct llama_model *]
     * @param buf [char *]
     * @param bufSize [size_t]
     * @return [int]
     */
    public native static int llamaModelDesc(long model, byte[] buf, long bufSize);

    /**
     * Returns the total size of all the tensors in the model in bytes
     * @param model [const struct llama_model *]
     * @return [uint64_t]
     */
    public native static long llamaModelSize(long model);

    /**
     * Returns the total number of parameters in the model
     * @param model [const struct llama_model *]
     * @return [uint64_t]
     */
    public native static long llamaModelNParams(long model);

    /**
     * Get a llama model tensor
     * @param model [const struct llama_model *] model
     * @param name [const char *] tensor name
     * @return [struct ggml_tensor *]
     */
    public native static long llamaGetModelTensor(long model, String name);

    /**
     * Returns 0 on success
     * @param fnameInp [const char *]
     * @param fnameOut [const char *]
     * @param params [const struct llama_model_quantize_params *]
     * @return [int]
     */
    public native static int llamaModelQuantize(String fnameInp, String fnameOut, LlamaModelQuantizeParams params);

    /**
     * Returns 0 on success
     * @param model [const struct llama_model *]
     * @param pathLora [const char *]
     * @param scale [float]
     * @param pathBaseModel [const char *]
     * @param nThreads [int]
     * @return [int]
     */
    public native static int llamaModelApplyLoraFromFile(long model, String pathLora, float scale, String pathBaseModel, int nThreads);

    /**
     * Information associated with an individual cell in the KV cache view.
     */
    public static class KvCacheViewCell {
        // The position for this cell. Takes KV cache shifts into account.
        // May be negative if the cell is not populated.
        public final int pos;

        public KvCacheViewCell(int pos) {
            this.pos = pos;
        }
    }

    /**
     * An updateable view of the KV cache.
     */
    public static class KvCacheView {

        // Number of KV cache cells. This will be the same as the context size.
        public final int nCells;

        // Maximum number of sequences that can exist in a cell. It's not an error
        // if there are more sequences in a cell than this value, however they will
        // not be visible in the view cells_sequences.
        public final int nMaxSeq;

        // Number of tokens in the cache. For example, if there are two populated
        // cells, the first with 1 sequence id in it and the second with 2 sequence
        // ids then you'll have 3 tokens.
        public final int tokenCount;

        // Number of populated cache cells.
        public final int usedCells;

        // Maximum contiguous empty slots in the cache.
        public final int maxContiguous;

        // Index to the start of the max_contiguous slot range. Can be negative
        // when cache is full.
        public final int maxContiguousIdx;

        // Information for an individual cell.
        public final long cells;

        // The sequences for each cell. There will be n_max_seq items per cell.
        public final long cellsSequences;

        public KvCacheView(int nCells, int nMaxSeq, int tokenCount, int usedCells, int maxContiguous,
                           int maxContiguousIdx, long cells, long cellsSequences) {
            this.nCells = nCells;
            this.nMaxSeq = nMaxSeq;
            this.tokenCount = tokenCount;
            this.usedCells = usedCells;
            this.maxContiguous = maxContiguous;
            this.maxContiguousIdx = maxContiguousIdx;
            this.cells = cells;
            this.cellsSequences = cellsSequences;
        }
    }

    /**
     * Create an empty KV cache view. (use only for debugging purposes)
     * @param ctx [const struct llama_context *]
     * @param nMaxSeq [int32_t]
     * @return [struct llama_kv_cache_view]
     */
    public native static KvCacheView llamaKvCacheViewInit(long ctx, int nMaxSeq);

    /**
     * Free a KV cache view. (use only for debugging purposes)
     * @param view [struct llama_kv_cache_view *]
     */
    public native static void llamaKvCacheViewFree(long view);

    /**
     * Update the KV cache view structure with the current state of the KV cache. (use only for debugging purposes)
     * @param ctx [const struct llama_context *]
     * @param view [struct llama_kv_cache_view *]
     */
    public native static void llamaKvCacheViewUpdate(long ctx, long view);

    /**
     * Returns the number of tokens in the KV cache (slow, use only for debug)
     * @param ctx [const struct llama_context *]
     * @return [int]
     */
    public native static int llamaGetKvCacheTokenCount(long ctx);

    /**
     * Returns the number of used KV cells (i.e. have at least one sequence assigned to them)
     * @param ctx [const struct llama_context *]
     * @return [int]
     */
    public native static int llamaGetKvCacheUsedCells(long ctx);

    /**
     * Clear the KV cache
     * @param ctx [const struct llama_context *]
     */
    public native static void llamaKvCacheClear(long ctx);

    /**
     * Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
     * seq_id < 0 : match any sequence
     * p0 < 0     : [0,  p1]
     * p1 < 0     : [p0, inf)
     * @param ctx [const struct llama_context *]
     * @param seqId [llama_seq_id]
     * @param p0 [llama_pos]
     * @param p1 [llama_pos]
     */
    public native static void llamaKvCacheSeqRm(long ctx, int seqId, int p0, int p1);

    /**
     * Copy all tokens that belong to the specified sequence to another sequence
     * Note that this does not allocate extra KV cache memory - it simply assigns the tokens to the new sequence
     * p0 < 0 : [0,  p1]
     * p1 < 0 : [p0, inf)
     * @param ctx [const struct llama_context *]
     * @param seqIdSrc [llama_seq_id]
     * @param seqIdDst [llama_seq_id]
     * @param p0 [llama_pos]
     * @param p1 [llama_pos]
     */
    public native static void llamaKvCacheSeqCp(long ctx, int seqIdSrc, int seqIdDst, int p0, int p1);

    /**
     * Removes all tokens that do not belong to the specified sequence
     * @param ctx [const struct llama_context *]
     * @param seqId [llama_seq_id]
     */
    public native static void llamaKvCacheSeqKeep(long ctx, int seqId);

    /**
     * Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
     * If the KV cache is RoPEd, the KV data is updated accordingly
     * p0 < 0 : [0,  p1]
     * p1 < 0 : [p0, inf)
     * @param ctx [const struct llama_context *]
     * @param seqId [llama_seq_id]
     * @param p0 [llama_pos]
     * @param p1 [llama_pos]
     * @param delta [llama_pos]
     */
    public native static void llamaKvCacheSeqShift(long ctx, int seqId, int p0, int p1, int delta);

    //
    //  State / sessions
    //

    /**
     *  Returns the maximum size in bytes of the state (rng, logits, embedding
     *  and kv_cache) - will often be smaller after compacting tokens
     * @param ctx [const struct llama_context *]
     * @return [size_t]
     */
    public native static long llamaGetStateSize(long ctx);

    /**
     * Copies the state to the specified destination address.
     * Destination needs to have allocated enough memory.
     * Returns the number of bytes copied
     * @param ctx [const struct llama_context *]
     * @param dst [uint8_t *]
     * @return [size_t]
     */
    public native static long llamaCopyStateData(long ctx, byte[] dst);

    /**
     * Set the state reading from the specified address
     * @param ctx [const struct llama_context *]
     * @param src [uint8_t *]
     * @return [size_t]
     */
    public native static long llamaSetStateData(long ctx, byte[] src);

    /**
     * Load a session file
     * @param ctx [struct llama_context *]
     * @param pathSession [const char *]
     * @param tokensOut [llama_token *]
     * @param nTokenCapacity [size_t]
     * @param nTokenCountOut [size_t *]
     * @return [bool]
     */
    public native static boolean llamaLoadSessionFile(
            long ctx, String pathSession, long tokensOut, long nTokenCapacity, long nTokenCountOut);

    /**
     * Save a session file
     * @param ctx [struct llama_context *]
     * @param pathSession [const char *]
     * @param tokens [const llama_token *]
     * @param nTokenCount [size_t]
     * @return [bool]
     */
    public native static boolean llamaSaveSessionFile(long ctx, String pathSession, long tokens, long nTokenCount);

    //
    // Decoding
    //

    /**
     * Return batch for single sequence of tokens starting at pos_0
     * NOTE: this is a helper function to facilitate transition to the new batch API - avoid using it
     * @param tokens [llama_token *]
     * @param nTokens [int32_t]
     * @param pos0 [llama_pos]
     * @param seqId [llama_seq_id]
     * @return [struct llama_batch]
     */
    public native static LlamaBatch llamaBatchGetOne(long tokens, int nTokens, int pos0, int seqId);

    /**
     * Allocates a batch of tokens on the heap that can hold a maximum of n_tokens
     * Each token can be assigned up to n_seq_max sequence ids
     * The batch has to be freed with llama_batch_free()
     * If embd != 0, llama_batch.embd will be allocated with size of n_tokens * embd * sizeof(float)
     * Otherwise, llama_batch.token will be allocated to store n_tokens llama_token
     * The rest of the llama_batch members are allocated with size n_tokens
     * All members are left uninitialized
     * @param nTokens [int32_t]
     * @param embd [int32_t]
     * @param nSeqMax [int32_t]
     * @return [struct llama_batch]
     */
    public native static LlamaBatch llamaBatchInit(int nTokens, int embd, int nSeqMax);

    /**
     * Frees a batch of tokens allocated with llama_batch_init()
     * @param batch [struct llama_batch]
     */
    public native static void llamaBatchFree(LlamaBatch batch);

    /**
     * Positive return values does not mean a fatal error, but rather a warning.
     *  0   - success
     *  1   - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
     *  < 0 - error
     * @param ctx [struct llama_context *]
     * @param batch [struct llama_batch]
     * @return [int]
     */
    public native static int llamaDecode(long ctx, LlamaBatch batch);

    /**
     * Set the number of threads used for decoding
     * @param ctx [struct llama_context *]
     * @param nThreads [uint32_t] the number of threads used for generation (single token)
     * @param nThreadsBatch [uint32_t] the number of threads used for prompt and batch processing (multiple tokens)
     */
    public native static void llamaSetNThreads(long ctx, int nThreads, int nThreadsBatch);

    /**
     * Token logits obtained from the last call to llama_eval()
     * The logits for the last token are stored in the last row
     * Logits for which llama_batch.logits[i] == 0 are undefined
     * Rows: n_tokens provided with llama_batch
     * Cols: n_vocab
     * @param ctx [struct llama_context *]
     * @return [float *]
     */
    public native static long llamaGetLogits(long ctx);

    /**
     * Logits for the ith token. Equivalent to:
     * llama_get_logits(ctx) + i*n_vocab
     * @param ctx [struct llama_context *]
     * @param i [int32_t]
     * @return [float *]
     */
    public native static long llamaGetLogitsIth(long ctx, int i);

    /**
     * Get the embeddings for the input
     * shape: [n_embd] (1-dimensional)
     * @param ctx [struct llama_context *]
     * @return [float *]
     */
    public native static long llamaGetEmbeddings(long ctx);

    //
    // Vocab
    //

    /**
     * @param model [const struct llama_model *]
     * @param token [llama_token]
     * @return [char *]
     */
    public native static String llamaTokenGetText(long model, int token);

    /**
     * @param model [const struct llama_model *]
     * @param token [llama_token]
     * @return [float]
     */
    public native static float llamaTokenGetScore(long model, int token);

    /**
     * @param model [const struct llama_model *]
     * @param token [llama_token]
     * @return [enum llama_token_type]
     */
    public native static int llamaTokenGetType(long model, int token);

    /**
     * Special tokens - beginning-of-sentence
     * @param model [const struct llama_model *]
     * @return [llama_token]
     */
    public native static int llamaTokenBos(long model);

    /**
     * Special tokens - end-of-sentence
     * @param model [const struct llama_model *]
     * @return [llama_token]
     */
    public native static int llamaTokenEos(long model);

    /**
     * @param model [const struct llama_model *]
     * @return [llama_token]
     */
    public native static int llamaTokenNl(long model);

    /**
     * Returns -1 if unknown, 1 for true or 0 for false.
     * @param model [const struct llama_model * model]
     * @return [int]
     */
    public native static int llamaAddBosToken(long model);

    /**
     * Returns -1 if unknown, 1 for true or 0 for false.
     * @param model [const struct llama_model *]
     * @return [int]
     */
    public native static int llamaAddEosToken(long model);

    // codellama infill tokens

    /**
     * Beginning of infill prefix
     * @param model [const struct llama_model *]
     * @return [llama_token]
     */
    public native static int llamaTokenPrefix(long model);

    /**
     * Beginning of infill middle
     * @param model [const struct llama_model *]
     * @return [llama_token]
     */
    public native static int llamaTokenMiddle(long model);

    /**
     * Beginning of infill suffix
     * @param model [const struct llama_model *]
     * @return [llama_token]
     */
    public native static int llamaTokenSuffix(long model);

    /**
     * End of infill middle
     * @param model [const struct llama_model *]
     * @return [llama_token]
     */
    public native static int llamaTokenEot(long model);

    //
    // Tokenization
    //

    /**
     * Convert the provided text into tokens.
     * Does not insert a leading space.
     * @param model [const struct llama_model *]
     * @param text [const char *]
     * @param textLen [int]
     * @param tokens [llama_token *] The tokens pointer must be large enough to hold the resulting tokens.
     * @param nMaxTokens [int]
     * @param addBos [bool] Add a beginning-of-sentence token at the beginning of the token sequence.
     * @param special [bool] Allow tokenizing special and/or control tokens which otherwise are not
     *                exposed and treated as plaintext.
     * @return Returns the number of tokens on success, no more than n_max_tokens
     *         Returns a negative number on failure - the number of tokens that would have been returned
     */
    public native static int llamaTokenize(long model, String text, int textLen, long tokens, int nMaxTokens, boolean addBos, boolean special);

    /**
     * Token Id -> Piece.
     * Uses the vocabulary in the provided context.
     * Does not write null terminator to the buffer.
     * User code is responsible to remove the leading whitespace of the
     * first non-BOS token when decoding multiple tokens.
     * @param model [const struct llama_model *]
     * @param token [llama_token]
     * @param buf [char *]
     * @param length [int]
     * @return [int]
     */
    public native static int llamaTokenToPiece(long model, int token, byte[] buf, int length);

    //
    // Grammar
    //

    /**
     * @param rules [const llama_grammar_element **]
     * @param nRules [size_t]
     * @param startRuleIndex [size_t]
     * @return [struct llama_grammar *]
     */
    public native static long llamaGrammarInit(long rules, long nRules, long startRuleIndex);

    /**
     * @param grammar [struct llama_grammar *]
     */
    public native static void llamaGrammarFree(long grammar);

    /**
     * @param grammar [const struct llama_grammar *]
     * @return [struct llama_grammar *]
     */
    public native static long llamaGrammarCopy(long grammar);

    //
    // Sampling functions
    //

    /**
     * Sets the current rng seed.
     * @param ctx [struct llama_context *]
     * @param seed [uint32_t]
     */
    public native static void llamaSetRngSeed(long ctx, int seed);

    /**
     * Repetition penalty described in <a href="https://arxiv.org/abs/1909.05858">CTRL academic paper</a>, with negative logit fix.
     * Frequency and presence penalties described in <a href="https://platform.openai.com/docs/api-reference/parameter-details">OpenAI API</a>.
     * @param ctx [struct llama_context *]
     * @param candidates [llama_token_data_array *]
     * @param lastTokens [const llama_token *]
     * @param penaltyLastN [size_t]
     * @param penaltyRepeat [float]
     * @param penaltyFreq [float]
     * @param penaltyPresent [float]
     */
    public native static void llamaSampleRepetitionPenalties(long ctx, long candidates, long lastTokens, long penaltyLastN, float penaltyRepeat, float penaltyFreq, float penaltyPresent);

    /**
     * Apply classifier-free guidance to the logits as described in academic paper
     * <a href="https://arxiv.org/abs/2306.17806">"Stay on topic with Classifier-Free Guidance"</a>
     * @param ctx [struct llama_context *]
     * @param candidates  [llama_token_data_array *] A vector of `llama_token_data` containing the candidate tokens,
     *                    the logits must be directly extracted from the original generation context without being sorted.
     * @param guidanceCtx [struct llama_context *] A separate context from the same model.
     *                    Other than a negative prompt at the beginning, it should have all generated
     *                    and user input tokens copied from the main context.
     * @param scale [float] Guidance strength. 1.0f means no guidance. Higher values mean stronger guidance.
     */
    public native static void llamaSampleClassifierFreeGuidance(long ctx, long candidates, long guidanceCtx, float scale);

    /**
     * Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits.
     * @param ctx [struct llama_context *]
     * @param candidates [llama_token_data_array *]
     */
    public native static void llamaSampleSoftmax(long ctx, long candidates);

    /**
     * Top-K sampling described in academic paper
     * <a href="https://arxiv.org/abs/1904.09751">"The Curious Case of Neural Text Degeneration"</a>
     * @param ctx [struct llama_context *]
     * @param candidates [llama_token_data_array *]
     * @param k [int]
     * @param minKeep [size_t]
     */
    public native static void llamaSampleTopK(long ctx, long candidates, int k, long minKeep);

    /**
     * Nucleus sampling described in academic paper in academic paper
     * <a href="https://arxiv.org/abs/1904.09751">"The Curious Case of Neural Text Degeneration"</a>
     * @param ctx [struct llama_context *]
     * @param candidates [llama_token_data_array *]
     * @param p [float]
     * @param minKeep [size_t]
     */
    public native static void llamaSampleTopP(long ctx, long candidates, float p, long minKeep);

    /**
     * Minimum P sampling as described in <a href="https://github.com/ggerganov/llama.cpp/pull/3841">...</a>
     * @param ctx [struct llama_context *]
     * @param candidates [llama_token_data_array *]
     * @param p [float]
     * @param minKeep [size_t]
     */
    public native static void llamaSampleMinP(long ctx, long candidates, float p, long minKeep);

    /**
     * Tail Free Sampling described in <a href="https://www.trentonbricken.com/Tail-Free-Sampling/">Tail-Free Sampling</a>.
     * @param ctx [struct llama_context *]
     * @param candidates [llama_token_data_array *]
     * @param z [float]
     * @param minKeep [size_t]
     */
    public native static void llamaSampleTailFree(long ctx, long candidates, float z, long minKeep);

    /**
     * Locally Typical Sampling implementation described in the paper
     * <a href="https://arxiv.org/abs/2202.00666">Locally Typical Sampling</a>.
     * @param ctx [struct llama_context *]
     * @param candidates [llama_token_data_array *]
     * @param p [float]
     * @param minKeep [size_t]
     */
    public native static void llamaSampleTypical(long ctx, long candidates, float p, long minKeep);

    /**
     * @param ctx [struct llama_context *]
     * @param candidates [llama_token_data_array *]
     * @param temp [float]
     */
    public native static void llamaSampleTemp(long ctx, long candidates, float temp);

    /**
     * Apply constraints from grammar
     * @param ctx [struct llama_context *]
     * @param candidates [llama_token_data_array *]
     * @param grammar [struct llama_grammar *]
     */
    public native static void llamaSampleGrammar(long ctx, long candidates, long grammar);

    /**
     * Mirostat 1.0 algorithm described in the paper <a href="https://arxiv.org/abs/2007.14966">Mirostat 1.0</a>.
     * Uses tokens instead of words.
     * @param ctx           [struct llama_context *]
     * @param candidates    [llama_token_data_array *] A vector of `llama_token_data` containing the candidate tokens,
     *                      their probabilities (p), and log-odds (logit) for the current position in the generated text.
     * @param tau           [float] The target cross-entropy (or surprise) value you want to achieve for the generated text.
     *                      A higher value corresponds to more surprising or less predictable text, while a lower value
     *                      corresponds to less surprising or more predictable text.
     * @param eta           [float] The learning rate used to update `mu` based on the error between the target and observed
     *                      surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly,
     *                      while a smaller learning rate will result in slower updates.
     * @param m             [int] The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that
     *                      is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they
     *                      use `m = 100`, but you can experiment with different values to see how it affects the performance
     *                      of the algorithm.
     * @param mu            [float *] Maximum cross-entropy. This value is initialized to be twice the target cross-entropy
     *                      (`2 * tau`) and is updated in the algorithm based on the error between the target and observed
     *                      surprisal.
     * @return              [llama_token]
     */
    public native static int llamaSampleTokenMirostat(long ctx, long candidates, float tau, float eta, int m, long mu);

    /**
     * Mirostat 2.0 algorithm described in the paper <a href="https://arxiv.org/abs/2007.14966">Mirostat 2.0</a>.
     * @param ctx [struct llama_context *]
     * @param candidates [llama_token_data_array *] A vector of `llama_token_data` containing the candidate tokens,
     * @param tau [float] The target cross-entropy (or surprise) value you want to achieve for the generated text.
     * @param eta [float] The learning rate used to update `mu` based on the error between the target and observed
     * @param mu [float *] Maximum cross-entropy. This value is initialized to be twice the target cross-entropy
     * @return [llama_token]
     */
    public native static int llamaSampleTokenMirostatV2(long ctx, long candidates, float tau, float eta, long mu);

    /**
     * Selects the token with the highest probability.
     * @param ctx [struct llama_context *]
     * @param candidates [llama_token_data_array *]
     * @return [llama_token]
     */
    public native static int llamaSampleTokenGreedy(long ctx, long candidates);

    /**
     * Randomly selects a token from the candidates based on their probabilities.
     * @param ctx [struct llama_context *]
     * @param candidates [llama_token_data_array *]
     * @return [llama_token]
     */
    public native static int llamaSampleToken(long ctx, long candidates);

    /**
     * Accepts the sampled token into the grammar
     * @param ctx [struct llama_context *]
     * @param grammar [struct llama_grammar *]
     * @param token [llama_token]
     */
    public native static void llamaGrammarAcceptToken(long ctx, long grammar, int token);

    //
    // Beam search
    //

    public static class BeamView {
        public final long tokens;
        public final long nTokens;
        public final float p;       // Cumulative beam probability (renormalized relative to all beams)
        public final boolean eob;   // Callback should set this to true when a beam is at end-of-beam.

        public BeamView(long tokens, long nTokens, float p, boolean eob) {
            this.tokens = tokens;
            this.nTokens = nTokens;
            this.p = p;
            this.eob = eob;
        }
    }

    /**
     * Passed to beam_search_callback function.
     * Whenever 0 < common_prefix_length, this number of tokens should be copied from any of the beams
     * (e.g. beams[0]) as they will be removed (shifted) from all beams in all subsequent callbacks.
     * These pointers are valid only during the synchronous callback, so should not be saved.
     */
    public static class BeamsState {
        public final long beamViews;
        public final long nBeams;               // Number of elements in beam_views[].
        public final long commonPrefixLength;   // Current max length of prefix tokens shared by all beams.
        public final boolean lastCall;          // True iff this is the last callback invocation.

        public BeamsState(long beamViews, long nBeams, long commonPrefixLength, boolean lastCall) {
            this.beamViews = beamViews;
            this.nBeams = nBeams;
            this.commonPrefixLength = commonPrefixLength;
            this.lastCall = lastCall;
        }
    }

    // Type of pointer to the beam_search_callback function.
    // void* callback_data is any custom data passed to llama_beam_search, that is subsequently
    // passed back to beam_search_callback. This avoids having to use global variables in the callback.
    // typedef void (*llama_beam_search_callback_fn_t)(void * callback_data, struct llama_beams_state);

    /**
     * Deterministically returns entire sentence constructed by a beam search.
     * @param ctx [struct llama_context *] Pointer to the llama_context.
     * @param callback [llama_beam_search_callback_fn_t] Invoked for each iteration of the beam_search loop, passing in beams_state.
     * @param callbackData [void *] A pointer that is simply passed back to callback.
     * @param nBeams [size_t] Number of beams to use.
     * @param nPast [int] Number of tokens already evaluated.
     * @param nPredict [int] Maximum number of tokens to predict. EOS may occur earlier.
     */
    public native static void llamaBeamSearch(long ctx, long callback, long callbackData, long nBeams, int nPast, int nPredict);

    /**
     * Performance information
     * @param ctx [struct llama_context *]
     * @return [struct llama_timings]
     */
    public native static LlamaTimings llamaGetTimings(long ctx);

    /**
     * Reset performance information
     * @param ctx [struct llama_context *]
     */
    public native static void llamaPrintTimings(long ctx);

    /**
     * @param ctx [struct llama_context *]
     */
    public native static void llamaResetTimings(long ctx);

    /**
     * Print system information
     * @return [const char *]
     */
    public native static String llamaPrintSystemInfo();

    /**
     * Set callback for all future logging events.
     * If this is not called, or NULL is supplied, everything is output on stderr.
     * @param logCallback [ggml_log_callback]
     * @param userData [void *]
     */
    public native static void llamaLogSet(long logCallback, long userData);

    /**
     * Dump timing information in YAML format
     * @param stream [FILE *]
     * @param ctx [const struct llama_context *]
     */
    public native static void llamaDumpTimingInfoYaml(long stream, long ctx);
}
