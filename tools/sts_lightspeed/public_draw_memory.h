#pragma once

#include <functional>

// Observes executed public card movements only. No CardManager ABI changes.
namespace sts {
struct CardManager;
namespace public_draw {
enum class Event { top, removed, randomized };
using Observer = std::function<void(const CardManager *, Event, int)>;
inline thread_local const Observer *observer = nullptr;
inline void notify(const CardManager *cards, Event event, int id = -1) {
    if (observer) (*observer)(cards, event, id);
}
struct Scope {
    const Observer *previous;
    Observer current;
    explicit Scope(Observer callback) : previous(observer), current(std::move(callback)) {
        observer = &current;
    }
    ~Scope() { observer = previous; }
};
}
}
