#pragma once
#include <algorithm>
#include <atomic>
#include <concepts>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

template<class P>
concept MsfdProblem =
    std::default_initializable<P> &&
    requires(P& a, P& b) {
        { a.split(b) } -> std::same_as<void>;
        { a.can_split() } -> std::same_as<bool>;
        { a.is_solved() } -> std::same_as<bool>;
        { a.a_step_forward() } ->
            std::same_as<void>;
        { a.swap(b) } -> std::same_as<void>;
        { a.merge(b) } -> std::same_as<void>;
    };

// Two cache lines: also defeats the adjacent-line prefetcher.
inline constexpr std::size_t kMsfdAlign = 128;

// Kept at namespace scope: MSVC cannot see an enclosing function
// template's local enum from a local class (C2065).
namespace msfd_detail {

// hungry: out of work, waiting
// claimed: a donor is splitting into this slot
// work: slot filled, run it
// done: search over
enum State : int { hungry, claimed, work, done };

template<class P>
struct alignas(kMsfdAlign) Slot {
    alignas(kMsfdAlign) std::atomic<int> state{hungry};
    std::mutex mu;
    std::condition_variable cv;
    alignas(kMsfdAlign) P prob;

    // claimed -> work and hungry -> done go through here so
    // a waiter cannot miss them; hungry -> claimed need not,
    // since a waiter keeps sleeping on claimed anyway.
    void wake(int v) {
        {
            std::lock_guard<std::mutex> g(mu);
            state.store(v, std::memory_order_release);
        }
        cv.notify_one();
    }

    // Returns false once the search is over.
    bool wait_for_work() {
        auto ready = [&] {
            const int v = state.load(std::memory_order_acquire);
            return v == work || v == done;
        };
        if (!ready()) {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, ready);
        }
        return state.load(std::memory_order_relaxed) == work;
    }
};

struct alignas(kMsfdAlign) Counter {
    std::atomic<int> v{0};
};

}  // namespace msfd_detail

// Runs on dedicated threads, never a work-stealing pool: an
// out-of-work worker blocks until it is handed work, so all T
// workers must run concurrently.
template<MsfdProblem P>
void msfd(P& root, int n_threads, bool forward_first) {
    using enum std::memory_order;
    using enum msfd_detail::State;
    using Slot = msfd_detail::Slot<P>;
    using Counter = msfd_detail::Counter;
    const int T = std::clamp(n_threads, 1, std::max(1, int(
        std::thread::hardware_concurrency())));

    if (forward_first) root.a_step_forward();
    if (root.is_solved()) return;
    if (T == 1) {
        while (!root.is_solved())
            root.a_step_forward();
        return;
    }

    auto slots = std::make_unique<Slot[]>(size_t(T));
    // Worker x is busy or claimed iff it is not counted here, so
    // the count reaches T only when no work is left anywhere.
    auto n_hungry = std::make_unique<Counter>();
    n_hungry->v.store(T - 1, relaxed);
    std::mutex merge_mu;

    slots[0].prob.swap(root);
    slots[0].state.store(work, relaxed);

    auto donate = [&](P& me, int x) -> bool {
        if (!me.can_split()) return false;
        for (int y = x + 1; y != x + T; ++y) {
            Slot& s = slots[y < T ? y : y - T];
            int e = hungry;
            if (!s.state.compare_exchange_strong(
                    e, claimed, acquire, relaxed))
                continue;
            n_hungry->v.fetch_sub(1, relaxed);
            me.split(s.prob);
            s.wake(work);
            return true;
        }
        return false;
    };

    auto run = [&](int x) {
        Slot& s = slots[x];
        P& p = s.prob;
        while (s.wait_for_work()) {
            while (!p.is_solved())
                if (n_hungry->v.load(relaxed) <= 0
                    || !donate(p, x))
                    p.a_step_forward();
            {
                std::lock_guard<std::mutex> g(merge_mu);
                root.merge(p);
            }
            // Publish hungry before counting it, so a terminating
            // worker never has its done overwritten.
            s.state.store(hungry, release);
            if (n_hungry->v.fetch_add(1, acq_rel) + 1 == T) {
                for (int y = 0; y < T; ++y)
                    if (y != x) slots[y].wake(done);
                return;
            }
        }
    };

    std::vector<std::thread> ts;
    ts.reserve(size_t(T - 1));
    for (int i = 1; i < T; ++i)
        ts.emplace_back([&run, i] { run(i); });
    run(0);
    for (auto& t : ts) t.join();
}

template<MsfdProblem P>
void MultithreadSolverForDecomposableProblems(
    P& problem, int n_threads, bool forward_first) {
    msfd(problem, n_threads, forward_first);
}
