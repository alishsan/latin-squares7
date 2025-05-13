(ns latin-squares7.core-test
  (:require [clojure.test :refer :all]
    [clojure.spec.alpha :as s]
            [functions :as f]))

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
