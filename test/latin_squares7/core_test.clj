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
 
  (testing "Basic moves"
    (let [b (f/new-board)]
      (is (f/valid-move? b [ 0 0 1]))
      (is (not (f/valid-move? b [0 0 8]))) ; Now properly fails
      (let [b1 (f/make-move b [0 0 1])]
        (is (not (f/valid-move? b1 [0 1 1])))
        (is (not (f/valid-move? b1 [1 0 1])))
        (is (f/valid-move? b1 [ 1 1 2])))))

(deftest spec-validation-test
  (testing "Game specs"
    (is (s/valid? ::f/number 3))
    (is (not (s/valid? ::f/number 8)))
    (is (s/valid? ::f/move [0 0 1]))
    (is (not (s/valid? ::f/move [0 0 :x])))))

(deftest game-state-test
  (testing "Turn management"
    (let [g (f/new-game)]
      (is (= :alice (f/current-player g)))
      (let [g1 (f/make-move g [0 0 1])]
        (is (= :bob (f/current-player g1)))
        (is (= 1 (:turn-number g1))))))
)
