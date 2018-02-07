#ifndef _UNIVERSAL_MTL_H
#define _UNIVERSAL_MTL_H

#ifdef USE_ARM

typedef void (*task_func_t)(void*, int);

/**
 * Static Multi-Threads Library.
 */
typedef struct smtl_t* smtl_handle;

/**
 * Init smtl_handle.
 */
void smtl_init(smtl_handle *psh,
    int num_threads);

/**
 * Finalize the smtl_handle.
 */
void smtl_fini(smtl_handle sh);

/**
 * Get the threads number of this smtl.
 */
int smtl_get_num_threads(smtl_handle sh);

/**
 * Add a new task to smtl. This task will run
 * on thread whose id is (K mod num_threads),
 * where this task is the Kth task.
 */
void smtl_add_task(smtl_handle sh,
    task_func_t task_func,
    void *params);

/**
 * begin to run all tasks added before.
 */
void smtl_begin_tasks(smtl_handle sh);

/**
 * After call smtl_begin_tasks(),
 * call this api to wait all tasks finished.
 */
void smtl_wait_tasks_finished(smtl_handle sh);

/**
 * Dynamic Multi-Threads Library.
 */
typedef struct dmtl_t* dmtl_handle;

/**
 * Init the dmtl_handle.
 */
void dmtl_init(dmtl_handle *pdh,
    int num_threads);

/**
 * Finalize the dmtl_handle.
 */
void dmtl_fini(dmtl_handle dh);

/**
 * Get the threads number of this dmtl.
 */
int dmtl_get_num_threads(dmtl_handle dh);

/**
 * Add one task to dmtl. If there is an idle
 * thread, the task will be run immediately by
 * the idle thread. Otherwise, enqueue this
 * task, wait for a running thread to be idle.
 */
void dmtl_add_task(dmtl_handle dh,
    task_func_t task_func,
    void *params);

/**
 * Wait all tasks be finished.
 */
void dmtl_wait_tasks_finished(dmtl_handle dh);

#endif

#endif

