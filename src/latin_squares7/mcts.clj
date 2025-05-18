(ns latin-squares7.mcts
  (:require [functions :as f]
            [latin-squares7.nn :as nn]))

(defrecord Node [wins visits prior children])

(defn new-node
  "Creates a new node with default values"
  ([] (new-node 0.1))
  ([prior] (->Node 0.0 0 (or prior 0.1) {})))

(defn backpropagate [trie path result]
  (let [safe-result (double (or result 0.0))] ; Force result to be a double
    (loop [t trie
           p path
           r safe-result]
      (if (empty? p)
        t
        (let [current-path (pop p)
              node (or (get-in t current-path) (new-node))
              updated-node (-> node
                             (update :wins (fnil + 0.0) r) ; fnil provides default
                             (update :visits (fnil inc 0)))]
          (recur (assoc-in t current-path updated-node)
                 current-path
                 (- r)))))))


;; UCB1 formula for move selection
(defn ucb1 [node total-visits]
  (let [{:keys [wins visits prior]} node
        visits (double visits)
        exploitation (if (zero? visits) 
                       Double/POSITIVE_INFINITY 
                       (/ wins visits))
        exploration (* (Math/sqrt (/ (Math/log (inc total-visits)) 
                                    (inc visits)))
                      prior)]
    (+ exploitation exploration)))



(defn expand-node [trie path game-state]
  (let [{:keys [policy]} (nn/predict game-state)
        legal-moves (f/suggested-moves (:board game-state))]
    (reduce (fn [t move]
              (assoc-in t (conj path move)
                        (->Node 0 0 (get policy move 0.1) {})))
            trie
            legal-moves)))

(defn simulate [game-state]
  (loop [current-state game-state
         depth 0
         max-depth 100]
    (let [moves (f/suggested-moves (:board current-state))]
      (cond
        (f/game-over? current-state)
        (double (if (= :alice (f/current-player current-state)) -1.0 1.0))
        
        (>= depth max-depth)
        0.0
        
        (empty? moves)
        (double (if (= :alice (f/current-player current-state)) -1.0 1.0))
        
        :else
        (let [move (rand-nth moves)
              new-state (f/make-move current-state move)]
          (recur new-state (inc depth) max-depth))))))


(defn print-trie-stats [trie]
  (let [root (get-in trie [])]
    (println "Trie stats:")
    (println "Root visits:" (:visits root))
    (println "Top moves:"
      (->> (:children root)
           (sort-by (comp :visits second) >)
           (take 5)
           (map (fn [[k v]] [k (:visits v)]))))))



(defn select-node [trie path game-state]
  (let [node (get-in trie path)
        children (:children node)]
    (cond
      (empty? children) path
      (f/game-over? game-state) path
      :else (let [total-visits (reduce + (map :visits (vals children)))
                  [best-move _] (->> children
                                  (map (fn [[move child]]
                                         [move (ucb1 child total-visits)]))
                                  (apply max-key second))
                  new-state (or (f/make-move game-state best-move) game-state)]
              (select-node trie (conj path best-move) new-state)))))


(defn mcts [initial-game-state iterations]
  (loop [trie {[] (new-node)}
         i 0]
    (let [path (select-node trie [] initial-game-state)
          game-state (reduce (fn [gs p] 
                              (if (vector? p) 
                                (or (f/make-move gs p) gs)
                                gs))
                            initial-game-state
                            (filter vector? path))
          
          legal-moves (seq (f/suggested-moves (:board game-state)))
          
          ;; FIXED: Proper expansion check
          expanded-trie (if (and legal-moves
                                (not (get-in trie (conj path :children))))
                          (reduce (fn [t move]
                                    (assoc-in t (conj path move)
                                              (new-node (/ 1.0 (count legal-moves)))))
                                  trie
                                  (shuffle legal-moves))  ;; Shuffle for variety
                          trie)
          
          ;; FIXED: Proper simulation triggering
          result (if-let [children (get-in expanded-trie (conj path :children))]
                   (if (empty? children)
                     (double (if (= :alice (f/current-player game-state)) -1.0 1.0))
                     (simulate game-state))
                   (simulate game-state))]
      
      (when (zero? (mod i 100))
        (let [root-children (get-in trie [[] :children] {})]
          (println "Iteration" i "| Path length:" (count path)
                   "| Children:" (count root-children)
                   "| Result:" result)))
      
(when (zero? (mod i 1000))
  (print-trie-stats trie))

      (if (>= i iterations)
        (do (println "MCTS completed with" (count (get-in trie [[] :children])) "root children")
            trie)
        (recur (backpropagate expanded-trie path result) (inc i))))))


(defn best-move [trie game-state]
  (let [root-children (get-in trie [[] :children] {})]
    (when-not (empty? root-children)
      (->> root-children
           (sort-by (fn [[_ {:keys [visits wins]}]]
                      (/ wins (max 1 visits))))
           last
           first))))



(defn valid-node? [node]
  (and (map? node)
       (map? (:children node))
       (every? (fn [[_ v]] (contains? v :visits)) (:children node))))


(defn mcts-stats [trie]
  (let [root (get-in trie [[] :children] {})]
    {:total-visits (reduce + (map :visits (vals root)))
     :unique-moves (count root)
     :top-moves (->> root
                  (sort-by (comp :visits second) >)
                  (take 3)
                  (map (fn [[k v]] [k (:visits v)])))}))



