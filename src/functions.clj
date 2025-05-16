(ns functions
  (:require [clojure.spec.alpha :as s]
            [clojure.java.io :as io]))

;; ======================
;; Game Specifications
;; ======================
(s/def ::number (s/and int? #(<= 1 % 7)))
(s/def ::row-index (s/and int? #(<= 0 % 6)))
(s/def ::col-index (s/and int? #(<= 0 % 6)))
(s/def ::cell (s/nilable ::number))
(s/def ::row (s/coll-of ::cell :count 7))
(s/def ::board (s/coll-of ::row :count 7))
(s/def ::move (s/and vector?
                    #(= 3 (count %))
                    (s/cat :row ::row-index
                           :col ::col-index
                           :num ::number)))  ;; Strict number validation
;; ======================
;; Core Game Functions
;; ======================

(defn new-board [] 
  (vec (repeat 7 (vec (repeat 7 nil)))))

(defn get-row [board row]
  (filter some? (board row)))

(defn get-col [board col]
  (filter some? (map #(nth % col) board)))

(defn valid-number? [num]
  (and (integer? num)
       (<= 1 num 7)))

(defn valid-move? [board [row col num :as move]]
  (and (some? move)
       (<= 0 row 6)
       (<= 0 col 6)
       (<= 1 num 7)
       (nil? (get-in board [row col]))
       (not (some #{num} (get-row board row)))
       (not (some #{num} (get-col board col)))))

  (defn make-move [game-state move]
  (when (and game-state 
             move
             (valid-move? (:board game-state) move))
    (let [[row col num] move]
      (-> game-state
          (update :board #(assoc-in % [row col] num))
          (update :turn-number inc)))))
;; ======================
;; Game State Management
;; ======================
(defrecord GameState [board turn-number])

(defn new-game []
  (->GameState (new-board) 0))

(defn current-player [game-state]
  (if (even? (:turn-number game-state)) :alice :bob))


(defn make-move [game-state [row col num :as move]]
  (when (and game-state
             (s/valid? ::move move)
             (valid-move? (:board game-state) move))
    (-> game-state
        (update :board #(assoc-in % [row col] num))
        (update :turn-number inc))))

;; ======================
;; Test-Friendly Version
;; ======================
(defn make-move* [game-state move]
  {:pre [(or (nil? game-state) (s/valid? ::board (:board game-state)))
         (or (nil? move) (s/valid? ::move move))]}
  (make-move game-state move))
;; Add a board-safe version
;(defn make-move-board [board move]
;  (when (valid-move? board move)
;    (assoc-in board [row col] num)))

;; ======================
;; Move Analysis
;; ======================
(defn available-numbers [board]
  {:pre [(s/valid? ::board board)]}
  (let [used (set (filter some? (flatten board)))]
    (remove used (range 1 8))))

(defn suggested-moves [board]
  {:pre [(s/valid? ::board board)]}
  (->> (for [row (range 7)
             col (range 7)
             :when (nil? (get-in board [row col]))]
         (for [num (range 1 8)
               :when (valid-move? board [row col num])]
           [row col num]))
       (apply concat)
       (seq)))  ;; Returns nil if no moves available




(defn game-over? [game-state]
  {:pre [(s/valid? ::board (:board game-state))]}
  (or (every? some? (flatten (:board game-state)))
      (empty? (suggested-moves (:board game-state)))))

;; ======================
;; Serialization
;; ======================
(defn game-state->edn [game-state]
  {:pre [(instance? GameState game-state)]}
  {:board (:board game-state)
   :turn-number (:turn-number game-state)})

(defn edn->game-state [{:keys [board turn-number]}]
  (->GameState board turn-number))

(defn save-game [game-state path]
  (io/make-parents path)
  (spit path (pr-str (game-state->edn game-state))))

(defn load-game [path]
  (-> path slurp read-string edn->game-state))

;; ======================
;; Debugging Utilities
;; ======================
(defn explain-move [board move]
  (when-not (valid-move? board move)
    (println "Invalid move because:")
    (cond
      (not (s/valid? ::move move)) (do
                                    (println "- Move format is invalid")
                                    (s/explain ::move move))
      (some? (get-in board [(first move) (second move)]))
        (println "- Cell is already occupied")
      (some #{(last move)} (get-row board (first move)))
        (println "- Number exists in row")
      (some #{(last move)} (get-col board (second move)))
        (println "- Number exists in column"))))

(defn print-board [board]
  (doseq [row board]
    (println (mapv #(if (nil? %) "." %) row))))
