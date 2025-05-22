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
;; In functions.clj
(defn get-row [board row]
  (into #{} (filter some? (board row))))

(defn get-col [board col]
  (into #{} (filter some? (map #(nth % col) board))))

(defn new-board [] 
  (vec (repeat 7 (vec (repeat 7 nil)))))


(defn valid-number? [num]
  (and (integer? num)
       (<= 1 num 7)))

(defn valid-move? [board [row col num :as move]]
  (and (s/valid? ::move move)
       (nil? (get-in board [row col]))
       (not (contains? (set (get-row board row)) num))
       (not (contains? (set (get-col board col)) num))))


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
             (instance? GameState game-state)
             (s/valid? ::move move)
             (valid-move? (:board game-state) move))
    (-> game-state
        (update :board #(assoc-in % [row col] num))
        (update :turn-number inc))))


(defn valid-game-state? [game-state]
  (and (instance? GameState game-state)
       (s/valid? ::board (:board game-state))
       (integer? (:turn-number game-state))
       (>= (:turn-number game-state) 0)))

;; ======================
;; Test-Friendly Version
;; ======================
(defn make-move* [game-state move]
  {:pre [(or (nil? game-state) (s/valid? ::board (:board game-state)))
         (or (nil? move) (s/valid? ::move move))]}
  (if (and game-state move)
    (make-move game-state move)
    game-state))
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
  (let [empty-cells (for [row (range 7)
                         col (range 7)
                         :when (nil? (get-in board [row col]))]
                     [row col])
        available-nums (available-numbers board)]
    (->> empty-cells
         (mapcat (fn [[row col]]
                   (keep (fn [num]
                           (when (valid-move? board [row col num])
                             [row col num]))
                         available-nums)))
         (seq))))


(defn debug-move-generation [board]
  (println "\n=== DEBUGGING MOVE GENERATION ===")
  (println "Board validation:" (s/valid? ::board board))
  
  (let [empty-cells (for [row (range 7)
                        col (range 7)
                        :when (nil? (get-in board [row col]))]
                    [row col])]
    
    (println "Empty cells:" (count empty-cells))
    
    (doseq [[row col] (take 5 empty-cells)]
      (let [row-numbers (set (filter some? (board row)))
            col-numbers (set (filter some? (map #(nth % col) board)))
            available (remove (into row-numbers col-numbers) (range 1 8))]
        
        (println (format "Cell [%d %d] - blocked by row: %s, col: %s | can take: %s"
                         row col
                         (or row-numbers "none")
                         (or col-numbers "none")
                         (seq available))))))
)

(defn game-over? [game-state]
  {:pre [(s/valid? ::board (:board game-state))]}
  (or (every? some? (flatten (:board game-state)))  ;; Board is full
      (let [moves (suggested-moves (:board game-state))]
        (empty? moves))))  ;; No valid moves available

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

(defn print-board [board]
  (doseq [row board]
    (println (mapv #(if (nil? %) "." %) row))))
