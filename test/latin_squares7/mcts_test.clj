(ns latin-squares7.mcts-test
  (:require [clojure.test :refer :all]
            [latin-squares7.mcts :refer [mcts ucb1 select-path expand-node backpropagate simulate best-move get-node-stats inspect-trie]]
            [latin-squares7.functions :as f]))

;; Helper functions for testing
(defn create-test-board [rows]
  {:pre [(f/valid-game-state? (f/->GameState (vec (map vec rows)) 0))]}
  (f/->GameState (vec (map vec rows)) 0))

;; Test UCB1 calculation
(deftest ucb1-test
  (testing "UCB1 calculation"
    (let [node {:wins 5 :visits 10}
          parent-visits 100
          c 1.4]
      (is (number? (ucb1 node parent-visits c)))
      (is (pos? (ucb1 node parent-visits c))))))

;; Test node selection
(deftest select-path-test
  (testing "Path selection"
    (let [tree {:root {:wins 0 :visits 1 :children {}}}
          path (select-path tree)]
      (is (vector? path))
      (is (<= (count path) 10)))))

;; Test node expansion
(deftest expand-node-test
  (testing "Node expansion"
    (let [state (f/new-game)
          tree {:root {:wins 0 :visits 1 :children {} :state state}}
          path []
          expanded-path (expand-node tree path)]
      (is (vector? expanded-path))
      (is (pos? (count expanded-path))))))

;; Test simulation
(deftest simulate-test
  (testing "Simulation"
    (let [state (f/new-game)
          result (simulate state)]
      (is (number? result))
      (is (or (= result 0) (= result 1))))))

;; Test backpropagation
(deftest backpropagate-test
  (testing "Backpropagation"
    (let [tree (atom {:root {:wins 0 :visits 1 :children {}}})
          path []
          result 1
          _ (backpropagate tree path result)
          updated-tree @tree
          root (:root updated-tree)]
      (is (>= (:visits root) 2))
      (is (>= (:wins root) 0)))))

;; Test best move selection
(deftest best-move-test
  (testing "Best move selection"
    (let [valid-board (f/new-board)
          valid-tree {:root {:wins 0 
                           :visits 1 
                           :children {[0 0 1] {:wins 1 :visits 2 :children {}}
                                    [0 0 2] {:wins 0 :visits 1 :children {}}}}}
          move (best-move valid-tree)]
      (is (or (nil? move) (vector? move))))))

;; Test full MCTS process
(deftest mcts-full-test
  (testing "Full MCTS process"
    (let [state (f/new-game)
          iterations 100
          result (mcts state iterations)]
      (is (vector? result))
      (is (= 3 (count result)))
      (is (f/valid-move? (:board state) result)))))

;; Test edge cases
(deftest mcts-edge-cases-test
  (testing "MCTS edge cases"
    ;; Test with nil game state
    (is (nil? (best-move nil)))
    
    ;; Test with game over state
    (let [full-board (create-test-board [[1 2 3 4 5 6 7]
                                       [2 3 4 5 6 7 1]
                                       [3 4 5 6 7 1 2]
                                       [4 5 6 7 1 2 3]
                                       [5 6 7 1 2 3 4]
                                       [6 7 1 2 3 4 5]
                                       [7 1 2 3 4 5 6]])
          tree {:root {:wins 0 :visits 1 :children {} :state full-board}}]
      (is (nil? (best-move tree))))
    
    ;; Test with almost full board
    (let [almost-full-board (create-test-board [[1 2 3 4 5 6 7]
                                              [2 3 4 5 6 7 1]
                                              [3 4 5 6 7 1 2]
                                              [4 5 6 7 1 2 3]
                                              [5 6 7 1 2 3 4]
                                              [6 7 1 2 3 4 5]
                                              [7 1 2 3 4 5 nil]])
          tree {:root {:wins 0 
                      :visits 1 
                      :children {[6 6 6] {:wins 1 :visits 2 :children {}}
                               [6 6 7] {:wins 0 :visits 1 :children {}}}
                      :state almost-full-board}}]
      (let [move (best-move tree)]
        (is (vector? move))
        (is (= 3 (count move)))
        (is (f/valid-move? (:board almost-full-board) move))))))