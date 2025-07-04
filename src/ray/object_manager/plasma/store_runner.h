// Copyright 2025 The Ray Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <memory>
#include <string>

#include "absl/synchronization/mutex.h"
#include "ray/common/asio/instrumented_io_context.h"
#include "ray/common/file_system_monitor.h"
#include "ray/object_manager/plasma/plasma_allocator.h"
#include "ray/object_manager/plasma/store.h"

namespace plasma {

class PlasmaStoreRunner {
 public:
  PlasmaStoreRunner(std::string socket_name,
                    int64_t system_memory,
                    bool hugepages_enabled,
                    std::string plasma_directory,
                    std::string fallback_directory);
  void Start(ray::SpillObjectsCallback spill_objects_callback,
             std::function<void()> object_store_full_callback,
             ray::AddObjectCallback add_object_callback,
             ray::DeleteObjectCallback delete_object_callback);
  void Stop();

  bool IsPlasmaObjectSpillable(const ObjectID &object_id);

  int64_t GetConsumedBytes();

  int64_t GetCumulativeCreatedObjects() const {
    return store_->GetCumulativeCreatedObjects();
  }

  int64_t GetCumulativeCreatedBytes() const {
    return store_->GetCumulativeCreatedBytes();
  }

  int64_t GetFallbackAllocated() const;

  void GetAvailableMemoryAsync(std::function<void(size_t)> callback) const {
    main_service_.post([this, callback]() { callback(store_->GetAvailableMemory()); },
                       "PlasmaStoreRunner.GetAvailableMemory");
  }

 private:
  void Shutdown();
  mutable absl::Mutex store_runner_mutex_;
  std::string socket_name_;
  int64_t system_memory_;
  bool hugepages_enabled_;
  std::string plasma_directory_;
  std::string fallback_directory_;
  mutable instrumented_io_context main_service_{/*enable_lag_probe=*/false,
                                                /*running_on_single_thread=*/true};
  std::unique_ptr<PlasmaAllocator> allocator_;
  std::unique_ptr<ray::FileSystemMonitor> fs_monitor_;
  std::unique_ptr<PlasmaStore> store_;
};

// We use a global variable for Plasma Store instance here because:
// 1) There is only one plasma store thread in Raylet.
// 2) The thirdparty dlmalloc library cannot be contained in a local variable,
//    so even we use a local variable for plasma store, it does not provide
//    better isolation.
extern std::unique_ptr<PlasmaStoreRunner> plasma_store_runner;

}  // namespace plasma
