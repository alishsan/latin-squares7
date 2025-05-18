(ns latin-squares7.nn)

;; Simple neural network representation
(defrecord Network [input-shape policy-weights value-weights])

(defn build-network []
  (->Network
    [7 7 8]  ; Input shape
    (vec (repeatedly (* 7 7 8 7) rand))  ; Random policy weights
    (vec (repeatedly (* 7 7 8) rand)))    ; Random value weights
)

;; Simple network that returns both policy and value predictions
(defn predict [game-state]
  (let [board (:board game-state)
        ;; Dummy implementation - replace with real logic
        legal-moves (filter some? (flatten board))
        move-count (count legal-moves)
        prior-prob (/ 1.0 (max 1 move-count))]
    {:policy (zipmap legal-moves (repeat prior-prob))  ; Uniform priors
     :value (rand)}))



