// Stub implementation of server-models for Android API < 28 with GGML_DLCU
// When building with GGML_DLCU on Android API 24-27, the posix_spawn functions
// are not available, so we disable multi-model server features.

#include "server-models.h"
#include "server-common.h"
#include "preset.h"
#include "arg.h"

#include <functional>
#include <stdexcept>
#include <string>

static const char * const MULTIMODEL_ERROR_MSG =
    "Multi-model server features are not available on Android API < 28 when using GGML_DLCU.\n"
    "Reason: The subprocess library requires posix_spawn functions which are only available in Android API 28+.\n"
    "Options:\n"
    "  1. Upgrade to Android API 28+ (Android 9.0+)\n"
    "  2. Use single-model mode by specifying a model path with --model\n"
    "  3. Build without GGML_DLCU (if applicable)\n";

server_models::server_models(const common_params & params, int argc, char ** argv, char ** envp)
    : ctx_preset(LLAMA_EXAMPLE_SERVER),
      base_params(params),
      bin_path(argv[0]),
      base_env(),
      base_preset() {
    // Store environment variables
    for (char ** env = envp; *env != nullptr; env++) {
        base_env.push_back(std::string(*env));
    }
    // Load preset after environment is ready
    const_cast<common_preset&>(base_preset) = ctx_preset.load_from_args(argc, argv);
}

void server_models::load_models() {
    // Empty implementation - models are loaded explicitly via load()
}

bool server_models::has_model(const std::string & name) {
    (void)name;
    return false;
}

std::optional<server_model_meta> server_models::get_meta(const std::string & name) {
    (void)name;
    return std::nullopt;
}

std::vector<server_model_meta> server_models::get_all_meta() {
    return std::vector<server_model_meta>();
}

void server_models::load(const std::string & name) {
    (void)name;
    throw std::runtime_error(MULTIMODEL_ERROR_MSG);
}

void server_models::unload(const std::string & name) {
    (void)name;
    throw std::runtime_error(MULTIMODEL_ERROR_MSG);
}

void server_models::unload_all() {
    throw std::runtime_error(MULTIMODEL_ERROR_MSG);
}

void server_models::update_status(const std::string & name, server_model_status status, int exit_code) {
    (void)name;
    (void)status;
    (void)exit_code;
    // No-op in stub implementation
}

void server_models::wait_until_loaded(const std::string & name) {
    (void)name;
    throw std::runtime_error(MULTIMODEL_ERROR_MSG);
}

bool server_models::ensure_model_loaded(const std::string & name) {
    (void)name;
    throw std::runtime_error(MULTIMODEL_ERROR_MSG);
}

server_http_res_ptr server_models::proxy_request(const server_http_req & req, const std::string & method, const std::string & name, bool update_last_used) {
    (void)req;
    (void)method;
    (void)name;
    (void)update_last_used;
    throw std::runtime_error(MULTIMODEL_ERROR_MSG);
}

std::thread server_models::setup_child_server(const std::function<void(int)> & shutdown_handler) {
    (void)shutdown_handler;
    throw std::runtime_error(MULTIMODEL_ERROR_MSG);
}

// Private methods

void server_models::update_meta(const std::string & name, const server_model_meta & meta) {
    (void)name;
    (void)meta;
    throw std::runtime_error(MULTIMODEL_ERROR_MSG);
}

void server_models::unload_lru() {
    throw std::runtime_error(MULTIMODEL_ERROR_MSG);
}

void server_models::add_model(server_model_meta && meta) {
    (void)meta;
    throw std::runtime_error(MULTIMODEL_ERROR_MSG);
}
