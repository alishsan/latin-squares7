(ns latin-squares7.mcts-test
  (:require [clojure.test :refer :all]
            [functions :as f]
            [latin-squares7.mcts :as mcts]))

;; Helper functions for testing
(defn create-test-board [rows]
  {:pre [(f/valid-game-state? (f/->GameState (vec (map vec rows)) 0))]}
  (f/->GameState (vec (map vec rows)) 0))

(defn get-node-stats [trie path]
  (let [node (get-in trie path (mcts/new-node))]
    {:wins (:wins node)
     :visits (:visits node)
     :children-count (count (:children node))}))

;; Test UCB1 calculation
(deftest ucb1-test
  (testing "UCB1 formula properties"
    (let [node (mcts/new-node 0.1)
          total-visits 100]
      ;; Test unvisited node returns infinity
      (is (= Double/POSITIVE_INFINITY (mcts/ucb1 (mcts/new-node) total-visits)))
      
      ;; Test visited node calculation
      (let [visited-node (-> node
                            (assoc :wins 5.0)
                            (assoc :visits 10))]
        (is (number? (mcts/ucb1 visited-node total-visits)))
        (is (pos? (mcts/ucb1 visited-node total-visits)))))))

;; Test node selection
(deftest select-path-test
  (testing "Path selection properties"
    (let [game (f/new-game)
          trie {[] (mcts/new-node)}]
      
      ;; Test selection on empty board
      (let [path (mcts/select-path trie game)]
        (is (vector? path))
        (is (<= (count path) 10))  ;; Max depth is 10
        (is (every? #(f/valid-move? (:board game) %) path)))
      
      ;; Test selection with some moves played
      (let [game-with-moves (-> game
                               (f/make-move [0 0 1])
                               (f/make-move [1 1 2]))
            path (mcts/select-path trie game-with-moves)]
        (is (vector? path))
        (is (every? #(f/valid-move? (:board game-with-moves) %) path))))))

;; Test node expansion
(deftest expand-node-test
  (testing "Node expansion properties"
    (let [game (f/new-game)
          trie {[] (mcts/new-node)}
          path []
          expanded-trie (mcts/expand-node trie path game)]
      
      ;; Test root node expansion
      (let [root-node (get expanded-trie [])]
        (is (pos? (count (:children root-node))))
        (is (<= (count (:children root-node)) 50))  ;; Max 50 children
        (is (every? #(f/valid-move? (:board game) %) (keys (:children root-node)))))
      
      ;; Test expansion after some moves
      (let [game-with-moves (-> game
                               (f/make-move [0 0 1])
                               (f/make-move [1 1 2]))
            path [[0 0 1] [1 1 2]]
            expanded-trie (mcts/expand-node trie path game-with-moves)
            node (get-in expanded-trie path)]
        (is (pos? (count (:children node))))
        (is (every? #(f/valid-move? (:board game-with-moves) %) (keys (:children node))))))))

;; Test simulation
(deftest simulate-test
  (testing "Simulation properties"
    (let [game (f/new-game)]
      ;; Test simulation on empty board
      (let [result (mcts/simulate game)]
        (is (number? result))
        (is (<= -1.0 result 1.0)))
      
      ;; Test simulation on partially filled board
      (let [game-with-moves (-> game
                               (f/make-move [0 0 1])
                               (f/make-move [1 1 2]))
            result (mcts/simulate game-with-moves)]
        (is (number? result))
        (is (<= -1.0 result 1.0)))
      
      ;; Test simulation on full board
      (let [full-board (create-test-board [[1 2 3 4 5 6 7]
                                         [2 3 4 5 6 7 1]
                                         [3 4 5 6 7 1 2]
                                         [4 5 6 7 1 2 3]
                                         [5 6 7 1 2 3 4]
                                         [6 7 1 2 3 4 5]
                                         [7 1 2 3 4 5 6]])
            result (mcts/simulate full-board)]
        (is (number? result))
        (is (<= -1.0 result 1.0))))))

;; Test backpropagation
(deftest backpropagate-test
  (testing "Backpropagation properties"
    (let [game (f/new-game)
          trie {[] (mcts/new-node)}
          path [[0 0 1] [1 1 2]]
          result 1.0
          updated-trie (mcts/backpropagate trie path result)]
      
      ;; Test root node update
      (let [root-stats (get-node-stats updated-trie [])]
        (println "DEBUG: updated-trie after backpropagate:" updated-trie)
        (is (pos? (:visits root-stats)))
        (is (pos? (:wins root-stats))))
      
      ;; Test path node updates
      (let [first-move-stats (get-node-stats updated-trie [(first path)])
            second-move-stats (get-node-stats updated-trie path)]
        (is (pos? (:visits first-move-stats)))
        (is (pos? (:wins first-move-stats)))
        (is (pos? (:visits second-move-stats)))
        (is (pos? (:wins second-move-stats)))))))

;; Test full MCTS process
(deftest mcts-full-test
  (testing "Full MCTS process"
    (let [game (f/new-game)
          iterations 100
          trie (mcts/mcts game iterations)]
      
      ;; Test trie structure
      (is (map? trie))
      (is (pos? (count trie)))
      
      ;; Test root node
      (let [root-stats (get-node-stats trie [])]
        (is (>= (:visits root-stats) iterations))
        (is (pos? (:children-count root-stats))))
      
      ;; Test best move selection
      (let [move (mcts/best-move game iterations)]
        (is (vector? move))
        (is (= 3 (count move)))
        (is (f/valid-move? (:board game) move))))))

;; Test edge cases
(deftest mcts-edge-cases-test
  (testing "MCTS edge cases"
    ;; Test with nil game state
    (is (nil? (mcts/best-move nil 100)))
    
    ;; Test with game over state
    (let [full-board (create-test-board [[1 2 3 4 5 6 7]
                                       [2 3 4 5 6 7 1]
                                       [3 4 5 6 7 1 2]
                                       [4 5 6 7 1 2 3]
                                       [5 6 7 1 2 3 4]
                                       [6 7 1 2 3 4 5]
                                       [7 1 2 3 4 5 6]])]
      (is (nil? (mcts/best-move full-board 100))))
    
    ;; Test with almost full board
    (let [almost-full-board (create-test-board [[1 2 3 4 5 6 7]
                                              [2 3 4 5 6 7 1]
                                              [3 4 5 6 7 1 2]
                                              [4 5 6 7 1 2 3]
                                              [5 6 7 1 2 3 4]
                                              [6 7 1 2 3 4 5]
                                              [7 1 2 3 4 5 nil]])]
      (let [move (mcts/best-move almost-full-board 100)]
        (is (vector? move))
        (is (= 3 (count move)))
        (is (f/valid-move? (:board almost-full-board) move))))))