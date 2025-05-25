(ns latin-squares7.core-test
  (:require [clojure.test :refer :all]
    [clojure.spec.alpha :as s]
            [functions :as f]
            [latin-squares7.mcts :as mcts]))

(deftest game-logic-test
  (testing "Number validation"
    (is (f/valid-number? 1))
    (is (f/valid-number? 7))
    (is (not (f/valid-number? 0)))
    (is (not (f/valid-number? 8)))
    (is (not (f/valid-number? "1"))))
 )

(deftest move-validation-test
  (testing "Edge cases"
    (is (nil? (f/make-move nil [0 0 1])))       ; nil game state
    (is (nil? (f/make-move (f/new-game) nil)))   ; nil move
        (is (nil? (f/make-move (f/new-game) [0 0 8]))))) ; invalid number

(deftest valid-moves-test
  (testing "Successful moves and conflicts"
    (let [initial (f/new-game)
          after-first (f/make-move initial [0 0 1])
          after-second (f/make-move after-first [1 1 2])]
      
      ;; Verify moves were applied
      (is (= 1 (get-in (:board after-first) [0 0])))
      (is (= 2 (get-in (:board after-second) [1 1])))
      
      ;; Verify move to occupied cell fails
      (is (nil? (f/make-move after-second [0 0 3])))
      
      ;; Verify number conflicts
      (is (nil? (f/make-move after-second [0 1 1]))) ; Same row
      (is (nil? (f/make-move after-second [1 0 2])))))) ; Same column

  

(deftest spec-validation-test
  (testing "Game specs"
    (is (s/valid? ::f/number 3))
    (is (not (s/valid? ::f/number 8)))
    (is (s/valid? ::f/move [0 0 1]))
    (is (not (s/valid? ::f/move [0 0 :x])))  ;; Now properly fails
    (is (not (s/valid? ::f/move [0 0 8])))))  ;; Also fails

(deftest game-state-test
  (testing "Turn management"
    (let [g (f/new-game)]
      (is (= :alice (f/current-player g)))
      (let [g1 (f/make-move g [0 0 1])]
        (is (= :bob (f/current-player g1)))
        (is (= 1 (:turn-number g1))))))
)




(deftest suggested-moves-test
  (testing "Move suggestion correctness"
    ;; Create a valid board according to specs
    (let [valid-board [[1 nil nil nil nil nil nil]
                       [nil 2 nil nil nil nil nil]
                       [nil nil nil nil nil nil nil]
                       [nil nil nil nil nil nil nil]
                       [nil nil nil nil nil nil nil]
                       [nil nil nil nil nil nil nil]
                       [nil nil nil nil nil nil nil]]]
      ;; First verify the board is valid
      (is (s/valid? ::f/board valid-board))
      
      ;; Now test suggestions
      (let [suggestions (f/suggested-moves valid-board)]
        ;; Should not suggest moves to occupied cells
        (is (not-any? #{[0 0 1]} suggestions))
        (is (not-any? #{[1 1 2]} suggestions))
        
        ;; Should not suggest numbers already in row/column
        (is (not-any? #{[0 1 1]} suggestions)) ; 1 in row 0
        (is (not-any? #{[1 0 2]} suggestions)) ; 2 in column 0
        
        ;; Should suggest valid moves
        (is (some #{[0 1 3]} suggestions))
        (is (some #{[2 2 4]} suggestions))))))

;; Updated helper function with better error reporting
(defn create-test-board [rows]
  {:pre [(do (println "Validating board:" (s/explain-str ::f/board rows))
         (s/valid? ::f/board rows))]}
  (-> (f/new-game)
      (assoc :board (vec (map vec rows)))))

(deftest game-over-test
  (testing "Game termination conditions"
    ;; Test empty board - game should continue
    (is (not (f/game-over? (f/new-game))))
    
    ;; Test partially filled board with available moves
    (let [partial-board (create-test-board [[1 nil nil nil nil nil nil]
                                           [nil 2 nil nil nil nil nil]
                                           [nil nil nil nil nil nil nil]
                                           [nil nil nil nil nil nil nil]
                                           [nil nil nil nil nil nil nil]
                                           [nil nil nil nil nil nil nil]
                                           [nil nil nil nil nil nil nil]])]
      (is (not (f/game-over? partial-board))))
    
    ;; Test completely filled valid board - game should end
    (let [full-board (create-test-board [[1 2 3 4 5 6 7]
                                        [2 3 4 5 6 7 1]
                                        [3 4 5 6 7 1 2]
                                        [4 5 6 7 1 2 3]
                                        [5 6 7 1 2 3 4]
                                        [6 7 1 2 3 4 5]
                                        [7 1 2 3 4 5 6]])]
      (is (f/game-over? full-board)))
    
    ;; Test board with no valid moves remaining
    (let [blocked-board (create-test-board [[1 2 3 4 5 6 7]
                                           [2 nil nil nil nil nil nil]
                                           [3 nil nil nil nil nil nil]
                                           [4 nil nil nil nil nil nil]
                                           [5 nil nil nil nil nil nil]
                                           [6 nil nil nil nil nil nil]
                                           [7 nil nil nil nil nil nil]])]
      (is (f/game-over? blocked-board)))))

(deftest best-move-test
  (testing "MCTS best-move returns a valid move on a new game"
    (let [game (f/new-game)
          trie (mcts/mcts game 100)
          move (mcts/best-move trie)]
      (is (vector? move))
      (is (= 3 (count move)))
      (is (f/valid-move? (:board game) move))))
  (testing "MCTS best-move returns a valid move after one move"
    (let [game (f/new-game)
          first-move [0 0 1]
          game-after-move (f/make-move game first-move)
          trie (mcts/mcts game-after-move 100)
          compressed-move (mcts/compress-move first-move)
          path [compressed-move]
          move (mcts/best-move trie path)]
      (is (vector? move))
      (is (= 3 (count move)))
      (is (f/valid-move? (:board game-after-move) move)))))






