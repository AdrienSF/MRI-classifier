import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from multiprocessing import Process, Queue
import mlflow

class ParallelKFold():
    def k_fold(self, n_splits, try_model, X, y):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)
        res_queue = Queue()
        kprocs = []
        warning_tag = {'WARNING': 'MLFlow autologging does not support parallel processing.'}
        with mlflow.start_run(tags=warning_tag, nested=True) as run:
        # with mlflow.start_run() as run:

            for train_index, test_index in kf.split(X):
                p = Process(target=self.try_model_wrapper, args=(res_queue, try_model, X[train_index], y[train_index], X[test_index], y[test_index]))
                p.start()
                kprocs.append(p)
                
            for i in range(len(kprocs)):
                kprocs[i].join()
                print(str(100*(i+1)/len(kprocs)) + "% folds complete")

            res = np.array(self.dump_queue(res_queue))
            mean_acc = np.mean(res)

            mlflow.log_metric('k-fold_acc', mean_acc)

        return mean_acc

    def try_model_wrapper(self, res_queue, try_model, X_train, y_trian, X_val, y_val):
        res_queue.put(try_model(X_train, y_trian, X_val, y_val))

    def dump_queue(self, queue):
        """
        Empties all pending items in a queue and returns them in a list.
        """
        queue.put('STOP')
        result = []

        for i in iter(queue.get, 'STOP'):
            result.append(i)

        return result