from scipy.stats import rankdata
import numpy as np

def RankAveragingEnsemble(prob_list , weights=None):
        """
    Perform rank-based ensembling on a list of class probability predictions.

    This method is useful when individual models may produce uncalibrated or
    poorly scaled probabilities (which is the case for our base models). Instead of directly averaging the probabilities
    (soft voting), we convert them to per-class ranks, aggregate those ranks,
    normalize them, and select the class with the highest aggregated rank.

    Parameters
    ----------
    prob_list : List[np.ndarray]
        A list of NumPy arrays with shape (N, C), where N is the number of samples
        and C is the number of classes. Each array contains the predicted probabilities
        from a different model.

    weights : List[float], optional
        A list of weights (same length as `prob_list`) indicating the relative importance
        of each model. If None, all models are treated equally.

    Returns
    -------
    rank_preds : np.ndarray
        An array of shape (N,) containing the predicted class indices for each sample
        based on the aggregated rank scores.
    """
        if not weights:
            weights = [1.0] * len(prob_list)
        
        N,C = prob_list[0].shape
        ranks = np.zeros((N,C))
        for w , p in zip(weights , prob_list):
            for c in range(C):
                ranks[:,c]+=  w  *  rankdata(p[:,c],method="average")

        ranks = ranks/sum(weights)
        ranks = ranks/ranks.sum(axis = 1,keepdims=True)
        rank_preds = np.argmax(ranks,axis = 1)
        return rank_preds