#include "memvanta/prefetcher.hpp"
namespace memvanta {
Prefetcher::Prefetcher(const TensorStore&s,TensorCache&c):store_(s),cache_(c){worker_=std::thread(&Prefetcher::loop,this);}
Prefetcher::~Prefetcher(){ stop(); }
void Prefetcher::rethrow_if_failed(){std::exception_ptr error;{std::lock_guard lk(mu_);error=error_;}if(error)std::rethrow_exception(error);}
void Prefetcher::request(std::uint32_t id){
  rethrow_if_failed();
  if(stop_.load()||id>=store_.count()||cache_.contains(id))return;
  {std::lock_guard lk(mu_);if(stop_.load())return;if(pending_.insert(id).second)q_.push_back(id);}
  cv_.notify_one();
}
void Prefetcher::stop(){stop_.store(true);cv_.notify_all();if(worker_.joinable())worker_.join();}
void Prefetcher::loop(){
  while(true){
    std::uint32_t id;
    {std::unique_lock lk(mu_);cv_.wait(lk,[&]{return stop_.load()||!q_.empty();});if(stop_.load())return;id=q_.front();q_.pop_front();}
    try{
      store_.prefetch(id);auto&s=store_.slice(id);cache_.insert_prefetched(id,store_.ptr(id),s.bytes);store_.release(id);
      std::lock_guard lk(mu_);pending_.erase(id);
    }catch(...){
      {std::lock_guard lk(mu_);if(!error_)error_=std::current_exception();pending_.erase(id);q_.clear();pending_.clear();}
      stop_.store(true);cv_.notify_all();return;
    }
  }
}
}
