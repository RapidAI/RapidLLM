#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace rapidllm {

class WorkerPool {
public:
    static WorkerPool& instance();
    static int default_threads();
    void set_threads(int n);
    int threads() const { return n_; }
    void parallel_for(int nitems, const std::function<void(int begin, int end)>& fn);

private:
    WorkerPool();
    ~WorkerPool();
    WorkerPool(const WorkerPool&) = delete;
    void resize(int n);
    void worker_loop(int id);

    int n_ = 0;
    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::function<void(int, int)> job_;
    std::atomic<int> next_{0};
    std::atomic<int> remaining_{0};
    int job_n_ = 0;
    int grain_ = 1;
    int epoch_ = 0;
    bool stop_ = false;
    bool done_ = true;
};

void set_num_threads(int n);
int num_threads();

} // namespace rapidllm
