#include "memvanta/worker_pool.hpp"
namespace memvanta {
WorkerPool::WorkerPool(unsigned threads):threads_(std::max(1u,threads)){
    if(threads_<=1)return;
    workers_.reserve(threads_-1);
    for(unsigned tid=1;tid<threads_;++tid)workers_.emplace_back([this,tid]{worker(tid);});
}
WorkerPool::~WorkerPool(){
    {std::lock_guard<std::mutex> lk(m_);stop_=true;++generation_;}
    cv_.notify_all();for(auto&t:workers_)t.join();
}
void WorkerPool::worker(unsigned tid){
    std::size_t seen=0;
    for(;;){
        std::function<void(std::size_t,std::size_t)> f;std::size_t n=0,g=0;
        {std::unique_lock<std::mutex> lk(m_);cv_.wait(lk,[&]{return stop_||generation_!=seen;});if(stop_)return;seen=g=generation_;f=fn_;n=n_;}
        const std::size_t step=(n+threads_-1)/threads_,a=tid*step,b=std::min(n,a+step);
        try{if(a<b)f(a,b);}catch(...){std::lock_guard<std::mutex> lk(m_);if(g==generation_&&!error_)error_=std::current_exception();}
        {std::lock_guard<std::mutex> lk(m_);if(g==generation_ && ++finished_==threads_-1)done_.notify_one();}
    }
}
void WorkerPool::parallel_for(std::size_t n,const std::function<void(std::size_t,std::size_t)>& fn){
    if(!n)return;
    std::lock_guard<std::mutex> call_lock(call_mu_);
    if(threads_<=1){fn(0,n);return;}
    {std::lock_guard<std::mutex> lk(m_);fn_=fn;n_=n;finished_=0;error_=nullptr;++generation_;}
    cv_.notify_all();
    const std::size_t step=(n+threads_-1)/threads_,b=std::min(n,step);
    try{if(b)fn(0,b);}catch(...){std::lock_guard<std::mutex> lk(m_);if(!error_)error_=std::current_exception();}
    std::unique_lock<std::mutex> lk(m_);done_.wait(lk,[&]{return finished_==threads_-1;});auto error=error_;lk.unlock();if(error)std::rethrow_exception(error);
}
} // namespace memvanta
