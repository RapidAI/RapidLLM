#include "rapidllm/runtime/thread_pool.h"

#include <algorithm>

namespace rapidllm {

int WorkerPool::default_threads() {
    unsigned n = std::thread::hardware_concurrency();
    return n ? static_cast<int>(n) : 4;
}

WorkerPool& WorkerPool::instance() {
    static WorkerPool p;
    return p;
}

WorkerPool::WorkerPool() { resize(default_threads()); }

WorkerPool::~WorkerPool() {
    {
        std::lock_guard<std::mutex> g(mu_);
        stop_ = true;
        ++epoch_;
    }
    cv_.notify_all();
    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }
}

void WorkerPool::set_threads(int n) {
    if (n < 1) n = 1;
    {
        std::lock_guard<std::mutex> g(mu_);
        if (n == n_ && !workers_.empty()) return;
        stop_ = true;
        ++epoch_;
    }
    cv_.notify_all();
    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }
    workers_.clear();
    stop_ = false;
    n_ = n;
    for (int i = 0; i < n_; ++i) workers_.emplace_back([this] { worker_loop(0); });
}

void WorkerPool::resize(int n) { set_threads(n); }

void WorkerPool::worker_loop(int) {
    int seen = 0;
    for (;;) {
        std::function<void(int, int)> job;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return stop_ || epoch_ != seen; });
            if (stop_) return;
            seen = epoch_;
            job = job_;
        }
        if (!job) continue;
        for (;;) {
            const int b = next_.fetch_add(grain_, std::memory_order_relaxed);
            if (b >= job_n_) break;
            job(b, std::min(b + grain_, job_n_));
        }
        if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard<std::mutex> g(mu_);
            done_ = true;
            cv_.notify_all();
        }
    }
}

void WorkerPool::parallel_for(int nitems, const std::function<void(int begin, int end)>& fn) {
    if (nitems <= 0) return;
    if (n_ <= 1 || nitems < 48) {
        fn(0, nitems);
        return;
    }
    {
        std::unique_lock<std::mutex> lk(mu_);
        job_ = fn;
        job_n_ = nitems;
        grain_ = std::max(8, nitems / (n_ * 4));
        next_.store(0, std::memory_order_relaxed);
        remaining_.store(n_, std::memory_order_relaxed);
        done_ = false;
        ++epoch_;
        cv_.notify_all();
        cv_.wait(lk, [&] { return done_; });
        job_ = nullptr;
    }
}

void set_num_threads(int n) { WorkerPool::instance().set_threads(n); }
int num_threads() { return WorkerPool::instance().threads(); }

} // namespace rapidllm
