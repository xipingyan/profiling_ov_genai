// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "load_image.hpp"
#include <openvino/genai/visual_language/pipeline.hpp>
#include <filesystem>
#include <chrono>

static bool print_subword(std::string &&subword)
{
    static int infer_id = 0;
    return !(std::cout << "infer_id:[" << infer_id++ << "]=" << subword << std::endl
                       << std::flush);
}

int test_llm_lookup(int argc, char* argv[]) {
    ov::AnyMap enable_compile_cache;
    std::string model_path = "../models/OpenVINO/Qwen2-0.5B-int8-ov/";

    std::cout << "== test_llm_lookup ..." << std::endl;
    std::cout << "   == Macro: PA=1     cfg[\"ATTENTION_BACKEND\"] = PA" << std::endl;
    std::cout << "   == Macro: SDPA=1   cfg[\"ATTENTION_BACKEND\"] = SDPA, Default SDPA." << std::endl;
    std::cout << "   == Macro: LOOKUP=1 cfg[\"prompt_lookup\"] = 1, Default 0." << std::endl;
    
    std::string device = "GPU.0";
    if (device == "GPU") {
        // Cache compiled models on disk for GPU to save time on the
        // next run. It's not beneficial for CPU.
        enable_compile_cache.insert({ov::cache_dir("vlm_cache")});
        std::cout << "    enable_compile_cache = " << "vlm_cache" << std::endl;
    }

    bool enable_look_up = false;
    if (std::getenv("LOOKUP") && std::string(std::getenv("LOOKUP")) == std::string("1")) {
        enable_look_up = true;
    }

    ov::AnyMap cfg;
    if (std::getenv("PA") && std::string(std::getenv("PA")) == std::string("1")) {
        cfg["ATTENTION_BACKEND"] = "PA";
    } else {
        cfg["ATTENTION_BACKEND"] = "SDPA";
    }
    if (enable_look_up) {
        cfg["prompt_lookup"] = true;
        std::cout << "  == cfg[\"prompt_lookup\"] = " << cfg["prompt_lookup"].as<bool>() << std::endl;
    }
    std::cout << "  == cfg[\"ATTENTION_BACKEND\"] = " << cfg["ATTENTION_BACKEND"].as<std::string>() << std::endl;

    ov::genai::LLMPipeline pipe(model_path, device, cfg);

    ov::genai::GenerationConfig config;
    config.max_new_tokens = 20;
    // config.num_beam_groups = 3;
    // config.num_beams = 15;
    // config.diversity_penalty = 1.0f;
    // config.num_return_sequences = config.num_beams;
    config.do_sample=false;
    config.temperature=0.1;

    if (enable_look_up)
    {
        // add parameter to enable prompt lookup decoding to generate `num_assistant_tokens` candidates per iteration
        config.num_assistant_tokens = 5;
        // Define max_ngram_size
        config.max_ngram_size = 3;
    }

    // config.apply_chat_template=true;
    // pipe.get_tokenizer().set_chat_template("{\"enable_thinking\": False}");

    // ov::genai::ChatHistory history({{{"role", "user"}, 
    //     {"content", "You are an AI assistant, please answer with json format."}}});
    // pipe.get_tokenizer().apply_chat_template(history, true);

    for (size_t i = 0;i < 3; i++) {
        ov::genai::ChatHistory history;
        history.push_back({{"role", "user"}, {"content", "What is the capital of China?"}});
        auto t1 = std::chrono::high_resolution_clock::now();
        auto outputs = pipe.generate(history, config);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "time:" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms, " << outputs << '\n';
    }
    return 0;
}

int test_vllm_lookup(int argc, char* argv[]) {
    std::cout << "== test_vllm_lookup ..." << std::endl;
    std::cout << "   == Macro: PA=1     cfg[\"ATTENTION_BACKEND\"] = PA" << std::endl;
    std::cout << "   == Macro: SDPA=1   cfg[\"ATTENTION_BACKEND\"] = SDPA, Default SDPA." << std::endl;
    std::cout << "   == Macro: LOOKUP=1 cfg[\"prompt_lookup\"] = 1, Default 0." << std::endl;
    std::cout << "   == Macro: DEV=GPU  Default GPU." << std::endl;

    ov::AnyMap enable_compile_cache;
    std::string model_path = "../models/ov/Qwen2.5-VL-3B-Instruct/INT4/";
    std::string device = "GPU.0";
    if (std::getenv("DEV"))
    {
        device = std::string(std::getenv("DEV"));
    }
    if (device == "GPU")
    {
        enable_compile_cache.insert({ov::cache_dir("vlm_cache")});
        std::cout << "    enable_compile_cache = " << "vlm_cache" << std::endl;
    }
    std::cout << "  == device = " << device << std::endl;

    bool enable_look_up = false;
    if (std::getenv("LOOKUP") && std::string(std::getenv("LOOKUP")) == std::string("1")) {
        enable_look_up = true;
    }
    std::cout << "  == enable_look_up = " << enable_look_up << std::endl;

    ov::genai::GenerationConfig config;
    config.max_new_tokens = 64;
    config.do_sample=false;
    config.temperature=0.1;
    if (enable_look_up)
    {
        // add parameter to enable prompt lookup decoding to generate `num_assistant_tokens` candidates per iteration
        config.num_assistant_tokens = 3;
        // Define max_ngram_size
        config.max_ngram_size = 3;
    }

    ov::AnyMap cfg;
    if (std::getenv("PA") && std::string(std::getenv("PA")) == std::string("1")) {
        cfg["ATTENTION_BACKEND"] = "PA";
    } else {
        cfg["ATTENTION_BACKEND"] = "SDPA";
    }
    if (enable_look_up) {
        cfg["prompt_lookup"] = enable_look_up;
        std::cout << "  == cfg[\"prompt_lookup\"] = " << cfg["prompt_lookup"].as<bool>() << std::endl;
    }

    std::cout << "  == cfg[\"ATTENTION_BACKEND\"] = " << cfg["ATTENTION_BACKEND"].as<std::string>() << std::endl;

    ov::genai::VLMPipeline pipe(model_path, device, cfg);

    auto images = utils::load_images("../test_video/cat_120_100.png");
    std::string prompts = "Is there animal in this image? please answer like: \"There is 2 ducks in this image.\"";
    prompts = "请描述图片";

    for (size_t i = 0;i < 1; i++) {
        // pipe.start_chat();
        auto t1 = std::chrono::high_resolution_clock::now();
        auto outputs = pipe.generate(prompts, ov::genai::image(images[0]), ov::genai::generation_config(config));
        //  ov::genai::streamer(print_subword));
        auto t2 = std::chrono::high_resolution_clock::now();
        // pipe.finish_chat();
        std::cout << " time:" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms, " << std::endl;
        std::cout << " outputs:" << outputs << std::endl;
        std::cout << " TTFT: " << outputs.perf_metrics.get_ttft().mean << " ± " << outputs.perf_metrics.get_ttft().std << " ms" << std::endl;
        std::cout << " TPOT: " << outputs.perf_metrics.get_tpot().mean << " ± " << outputs.perf_metrics.get_tpot().std << " ms" << std::endl;
        std::cout << " generated num: " << outputs.perf_metrics.get_num_generated_tokens() << std::endl;
        std::cout << " infer num: " << outputs.perf_metrics.raw_metrics.m_token_infer_durations.size() << std::endl;
    }
    return 0;
}