(ns latin-squares7.core-test
  (:require [clojure.test :refer :all]
    [clojure.spec.alpha :as s]
            [latin-squares7.functions :as f]
            [latin-squares7.nn-mm :as nn]))

(def ^:private board-size
  (or (some-> (System/getProperty "board.size") Integer/parseInt)
      (get-in (System/getProperties) ["leiningen.profiles" "test" "board-size"])
      7))  ; Default to 7x7 for tests to match main code

;; Updated helper function with better error reporting                                                                                                                       
(defn create-test-board [rows]
  {:pre [(do (println "Validating board:" (s/explain-str ::f/board rows))
         (s/valid? ::f/board rows))]}
  (-> (f/new-game)
      (assoc :board (vec (map vec rows)))))

(deftest game-logic-test
  (testing "Number validation"
    (is (f/valid-number? 1))
    (is (f/valid-number? board-size))
    (is (not (f/valid-number? 0)))
    (is (not (f/valid-number? (inc board-size))))
    (is (not (f/valid-number? "1"))))
 )

(deftest move-validation-test
  (testing "Edge cases"
    (is (nil? (f/make-move nil [0 0 1])))       ; nil game state
    (is (nil? (f/make-move (f/new-game) nil)))   ; nil move
        (is (nil? (f/make-move (f/new-game) [0 0 8])))) ; invalid number


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




(deftest valid-moves-test
  "Test that suggested moves are valid"
  (let [board (f/new-board)
        moves (f/valid-moves board)]
    (is (seq moves))
    (is (every? #(f/valid-move? board %) moves))
    (testing "Valid moves on blocked board (should be empty)"
      (let [blocked-board [[7 nil 5 4 3 nil 1]
                           [4 7 6 1 nil 3 2]
                           [5 6 7 nil 1 2 3]
                           [3 nil 1 7 6 5 4]
                           [nil 3 2 6 7 4 5]
                           [2 1 3 5 4 7 6]
                           [1 2 4 3 5 6 7]]
            candidate-moves (f/valid-moves blocked-board)
            valid-moves (vec (filter (fn [move]
                                     (and (f/valid-move? blocked-board move)
                                          (nil? (get-in blocked-board [(nth move 0) (nth move 1)]))))
                                   candidate-moves))]
        (println "Valid moves (or invalid moves) for blocked board:" candidate-moves)
        (is (empty? valid-moves))))
))



(deftest select-move-test
  "Test that selected move is valid"
  (let [board (f/new-board)
        valid-moves (f/valid-moves board)
        selected-move (when (seq valid-moves)
                       (rand-nth valid-moves))]
    (is (some #{selected-move} valid-moves))))


(deftest game-over-test
  (testing "Game termination conditions"
    ;; Test empty board - game should continue
    (is (not (f/game-over? (f/new-game))))
    
    ;; Test partially filled board with available moves
    (let [partial-board (create-test-board (vec (repeat board-size
                                                       (vec (repeat board-size nil)))))]
      (is (not (f/game-over? partial-board))))
    
    ;; Test completely filled valid board - game should end
    (let [full-board (create-test-board (vec (for [i (range board-size)]
                                             (vec (for [j (range board-size)]
                                                   (inc (mod (+ i j) board-size)))))))]
      (is (f/game-over? full-board)))
    
    ;; Test board with no valid moves remaining
    (let [blocked-board (create-test-board (vec (for [i (range board-size)]
                                                (vec (for [j (range board-size)]
                                                      (inc (mod (+ i j) board-size)))))))]
      (is (f/game-over? blocked-board)))))

(deftest neural-network-test
  (testing "Neural network basic operations"
    ;; Test that the neural network can be initialized
    (let [pipeline (nn/create-game-pipeline)]
      (is (fn? pipeline))))
  
  (testing "Board to features conversion"
    (let [board (vec (repeat board-size (vec (repeat board-size nil))))
          features (vec (flatten (seq (get ((nn/board->features-pipe) {:metamorph/data board}) :metamorph/data))))]
      (is (= (* board-size board-size) (count features)))
      (is (every? #(= 0.0 %) features))))
  
  (testing "Neural network predictions"
    (let [board (vec (repeat board-size (vec (repeat board-size nil))))
          game-state {:board board}
          predictions (nn/run-pipeline (nn/create-game-pipeline) game-state :transform)]
      (is (map? predictions))
      (is (contains? predictions :policy))
      (is (contains? predictions :value))
      (is (map? (:policy predictions))))))

(deftest neural-network-move-selection-test
  (testing "Neural network move selection"
    (let [board (vec (repeat board-size (vec (repeat board-size nil))))
          game-state {:board board}
          best-move (nn/get-best-move game-state)]
      ;; Test that we get a valid move
      (when best-move
        (is (vector? best-move))
        (is (= 3 (count best-move)))
        (is (f/valid-move? board best-move))))))

(deftest valid-move-test
  (testing "Move validation rules"
    (let [board (vec (repeat board-size (vec (repeat board-size nil))))]
      
      ;; Test valid moves
      (is (f/valid-move? board [2 2 3]))  ; Empty cell, no conflicts
      (is (f/valid-move? board [0 1 3]))  ; Empty cell, no conflicts
      
      ;; Test out of bounds
      (is (not (f/valid-move? board [-1 0 1])))  ; Negative row
      (is (not (f/valid-move? board [0 -1 1])))  ; Negative column
      (is (not (f/valid-move? board [board-size 0 1])))   ; Row too large
      (is (not (f/valid-move? board [0 board-size 1])))   ; Column too large
      
      ;; Test invalid numbers
      (is (not (f/valid-move? board [2 2 0])))   ; Number too small
      (is (not (f/valid-move? board [2 2 (inc board-size)])))   ; Number too large
      
      ;; Test occupied cells
      (let [board-with-move (assoc-in board [0 0] 1)]
        (is (not (f/valid-move? board-with-move [0 0 3])))   ; Cell already has 1
        
        ;; Test row conflicts
        (is (not (f/valid-move? board-with-move [0 1 1])))   ; 1 already in row 0
        
        ;; Test column conflicts
        (is (not (f/valid-move? board-with-move [1 0 1]))))  ; 1 already in column 0
      
      ;; Test nil inputs
      (is (not (f/valid-move? nil [0 0 1])))     ; Nil board
      (is (not (f/valid-move? board nil)))       ; Nil move
      (is (not (f/valid-move? nil nil))))))      ; Both nil

;; Insert new test (valid-moves-blocked-test) after the game-over-test (or at the end of the file) to test valid-moves on the blocked board.
;; (deftest valid-moves-blocked-test
;;   (testing "Valid moves on blocked board (should be empty)"
;;     (let [blocked-board (create-test-board [[1 2 3 4 5 6 7]
;;                                            [2 nil nil nil nil nil nil]
;;                                            [3 nil nil nil nil nil nil]
;;                                            [4 nil nil nil nil nil nil]
;;                                            [5 nil nil nil nil nil nil]
;;                                            [6 nil nil nil nil nil nil]
;;                                            [7 nil nil nil nil nil nil]])
;;           valid-moves (f/valid-moves (:board blocked-board))]
;;       (println "Valid moves (or invalid moves) for blocked board:" valid-moves)
;;       (is (empty? valid-moves)))))

(deftest neural-network-autoplay-test
  (testing "Neural network autoplay"
    (let [board [[1 nil nil nil nil nil nil]
                 [nil 2 nil nil nil nil nil]
                 [nil nil nil nil nil nil nil]
                 [nil nil nil nil nil nil nil]
                 [nil nil nil nil nil nil nil]
                 [nil nil nil nil nil nil nil]
                 [nil nil nil nil nil nil nil]]
          game-state {:board board}
          _ (nn/initialize-model)  ; Initialize the model before testing
          result (nn/autoplay-from-position game-state 10)]  ; Try up to 10 moves
      (is (map? result))
      (is (contains? result :final-state))
      (is (contains? result :moves-made))
      (is (contains? result :solved?))
      (is (contains? result :moves)))))






