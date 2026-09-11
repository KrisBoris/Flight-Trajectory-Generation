# Flight Trajectory Generation — Przewodnik po projekcie

*Angielska wersja tego dokumentu jest dostępna w [`guide.md`](guide.md).*

## 1. Czym zajmuje się ten projekt

Projekt planuje lot pojedynczego drona w misji poszukiwawczo-ratunkowej (ang. *search-and-rescue*, SAR). Obszar poszukiwań jest modelowany jako siatka komórek. Każda komórka ma przypisaną **wartość**: prawdopodobieństwo, że w tym miejscu znajduje się poszukiwana osoba. Przelot z jednej komórki do sąsiedniej kosztuje pewną ilość energii/czasu (**koszt**), wyliczoną na podstawie rzeczywistego terenu, nad którym dron by przeleciał — podejście w górę kosztuje więcej niż zejście w dół, a dłuższy skok kosztuje więcej niż krótki. Dron dysponuje stałym **budżetem** (maksymalnym łącznym kosztem, jaki może wydać) i w większości misji musi też zdążyć wrócić do punktu startu, zanim budżet się wyczerpie.

Zadanie rozwiązywane przez projekt brzmi zatem: **zaczynając od ustalonej komórki bazowej, znajdź trasę przez siatkę, która zbierze jak najwięcej wartości (prawdopodobieństwa), nie przekraczając dostępnego budżetu** (a jeśli jest to wymagane — pozostawiając wystarczająco budżetu na powrót do bazy). To klasyczny problem badań operacyjnych zwany **problemem orienteeringowym** (ang. [Orienteering Problem](https://en.wikipedia.org/wiki/Orienteering_problem)) — wybór opłacalnego *podzbioru* nagród do zebrania w ramach budżetu, a nie (jak w problemie komiwojażera) konieczność odwiedzenia wszystkiego.

Nie istnieje jeden „poprawny” sposób rozwiązania tego problemu jednocześnie dobrze i szybko — jest to problem NP-trudny, więc algorytm gwarantujący znalezienie prawdziwie najlepszej odpowiedzi może być wolny na dużej mapie, natomiast szybki algorytm może obiecać jedynie *dobrą* odpowiedź, niekoniecznie *najlepszą*. Ten projekt implementuje **16 różnych algorytmów**, które inaczej rozkładają ten kompromis — od najprostszych możliwych reguł zachłannych, aż po solver dający wynik matematycznie optymalny — tak, by można je było uruchamiać i porównywać na tym samym scenariuszu.

## 2. Podstawowe pojęcia

Zrozumienie kilku wspólnych elementów znacznie ułatwia lekturę opisów algorytmów w dalszej części.

- **Siatka.** Tablica `rows × cols` komórek. Każda komórka jest połączona z maksymalnie 8 sąsiadami (kierunki: góra, góra-prawo, prawo, dół-prawo, dół, dół-lewo, lewo, góra-lewo).
- **Wartość (`coordinates_values`).** Prawdopodobieństwo ukrycia poszukiwanej osoby w danej komórce, z zakresu od `0.0` do `Constants.MAX_PROBABILITY` (`10.0`). Komórka, dla której nikt nie ustawił jawnego prawdopodobieństwa, domyślnie ma wartość `Constants.DEFAULT_PROBABILITY` (`0.0`) — czyli „brak informacji”, a nie realny sygnał. Niemal każdy algorytm traktuje komórkę jako prawdziwy **cel** tylko wtedy, gdy jej wartość jest *powyżej* tego progu; nietknięte komórki tła nigdy nie są celowo poszukiwane.
- **Koszt / wagi (`WeightsGrid`).** Tablica `rows × cols × 8` przechowująca koszt ruchu z danej komórki w każdym z 8 kierunków. Jest budowana na podstawie rzeczywistych lokalnych współrzędnych terenu (`x, y, z` w metrach): ruch kosztuje stałą stawkę za metr odległości poziomej, plus karę za nachylenie proporcjonalną do `(zmiana wysokości)² / odległość pozioma` — podejście dolicza się do tego kosztu, zejście daje zniżkę (ograniczoną od dołu, tak by długie zejście nigdy nie było darmowe ani „opłacalne”). Dlatego ruch po przekątnej ani podejście nigdy nie są zakładane jako kosztujące tyle samo co krótki, płaski krok — model kosztów odzwierciedla rzeczywisty teren.
- **`blocked_mask`.** Boolowska siatka komórek zamkniętych dla lotów (strefy burzowe, przestrzeń zastrzeżona), na które żaden algorytm nigdy nie może wejść.
- **Budżet (`max_cost`) i `require_return_to_base`.** Górny limit łącznego kosztu całej misji. Gdy powrót jest wymagany, każdy algorytm przed każdym kolejnym ruchem rezerwuje wystarczającą ilość pozostałego budżetu na najtańszą możliwą trasę powrotną — dzięki temu dron nigdy nie może utknąć poza punktem, z którego nie da się już wrócić.
- **Komórka startowa.** Ustalona przez misję (np. baza zespołu ratowniczego). Każdy algorytm zaczyna właśnie tutaj; projekt sprawdza też przy wczytywaniu scenariusza, czy komórka startowa nie znajduje się przypadkiem na liście zablokowanych.
- **Cele / pula kandydatów.** Komórki, których wartość przekracza `Constants.DEFAULT_PROBABILITY`. Większość algorytmów w projekcie (wszystkie poza najprostszym, zachłannym „spacerowiczem”) ogranicza wybór celów właśnie do tej puli, zamiast rozważać dosłownie każdą komórkę siatki — to właśnie dzięki temu nie marnują wysiłku na porównywanie ze sobą bezwartościowych, pozbawionych informacji komórek tła.
- **Stan faktyczny a wcześniejsze przekonanie.** `searched_person_locations` (wraz z `probability`) to zawsze tylko *przekonanie* o tym, gdzie osoba może się znajdować — to właśnie to widzi i przeszukuje logika wyboru celów każdego algorytmu. `actual_person_locations` to osobna, opcjonalna lista miejsc, w których osoba (lub osoby) *naprawdę* się znajdują. Nie odgrywa żadnej roli w decyzjach żadnego algorytmu — dodaje jedynie jedną dodatkową, niezależną od algorytmu regułę zakończenia na wierzchu tego, co dany algorytm i tak by zrobił — patrz **Zakończenie po odnalezieniu wszystkich**, tuż przed sekcją 4.1.

## 3. Jak korzystać z projektu

### 3.1 Struktura projektu

```
src/
  coordinates_grid/        struktury danych siatki + generowanie terenu/masek
  pathfinding_algorithms/  każdy algorytm opisany w sekcji 4
  helpers/                 stałe, wczytywanie JSON, benchmarking
  gui/                     wizualizator matplotlib/PyQt5
  main.py                  przykładowe pojedyncze uruchomienie „od A do Z”
test_data/                 przykładowe scenariusze misji (JSON)
drone_data/                przykładowe konfiguracje drona (JSON)
requirements.txt           numpy, ortools, matplotlib, PyQt5
```

### 3.2 Opis misji (plik scenariusza JSON, `test_data/*.json`)

```json
{
  "terrain": {
    "rows": 100, "cols": 100,
    "altitude_range": [0.0, 20.0],
    "cell_size_meters": 5.0,
    "max_gradient": 0.001,
    "seed": 1
  },
  "start_location": {"row": 0, "col": 0},
  "searched_person_locations": [
    {"row": 3, "col": 5, "probability": 9.5}
  ],
  "blocked_cells": [
    {"row": 10, "col": 10}
  ],
  "actual_person_locations": [
    {"row": 3, "col": 5}
  ]
}
```

`terrain` opisuje rozmiar siatki oraz sposób generowania jej losowej rzeźby terenu (`max_gradient` ogranicza maksymalne nachylenie między dwiema sąsiednimi komórkami — jest to współczynnik nachylenia, nie procent; `seed` sprawia, że ta losowa rzeźba jest powtarzalna — ten sam plik scenariusza zawsze odtwarza dokładnie tę samą mapę). `start_location` to stały punkt startu/powrotu. `searched_person_locations` to rzeczywiste cele (wcześniejsze przekonanie, wraz z prawdopodobieństwami). `blocked_cells` jest opcjonalne i wylicza stałe komórki zamknięte dla lotów — **komórka startowa nigdy nie może się na tej liście znaleźć**; loader natychmiast zgłasza błąd, jeśli tak się stanie. `actual_person_locations` jest opcjonalne i wylicza, gdzie osoba (lub osoby) faktycznie się znajdują (stan faktyczny, bez prawdopodobieństwa, zero lub więcej wpisów) — patrz **Zakończenie po odnalezieniu wszystkich** przed sekcją 4.1.

### 3.3 Opis drona (`drone_data/*.json`)

```json
{
  "climb_cost_per_meter": 0.5,
  "descent_cost_per_meter": 0.3,
  "base_cost": 1.0,
  "max_cost": 5000.0,
  "require_return_to_base": true
}
```

`base_cost` to stała stawka kosztu przelotu za metr; `climb_cost_per_meter`/`descent_cost_per_meter` skalują karę za nachylenie; `max_cost` to budżet misji.

### 3.4 Uruchomienie pojedynczej misji

`src/main.py` wczytuje scenariusz i konfigurację drona, buduje siatkę, uruchamia jeden algorytm (`TrajectoryGenerator.find_best_path(..., algorithm="grasp")`), wypisuje jednolinijkowe podsumowanie i otwiera okno GUI z widokiem 2D z góry (oraz widokiem 3D terenu, jeśli wygenerowano współrzędne terenu), pokazującym mapę cieplną prawdopodobieństwa poszukiwań, wybraną trasę, kierunek lotu oraz ewentualne komórki zablokowane.

```bash
cd src
python main.py
```

Jako `algorithm=` można podać dowolny z 16 kluczy wymienionych w sekcji 4 — patrz słownik `PATHFINDING_ALGORITHMS` w pliku `pathfinding_algorithms/trajectory_generator.py`, będący źródłem prawdy dla pełnej listy.

### 3.5 Porównywanie algorytmów (`helpers/benchmark.py`)

`run_benchmark(...)` uruchamia wybrany zestaw algorytmów (domyślnie: wszystkie) na wybranym zestawie plików scenariuszy (domyślnie: wszystkie w `test_data/`), dowolną liczbę powtórzeń każdy, po czym wypisuje/zapisuje tabelę wyników (liczba kroków, zużyty koszt, % wykorzystanego budżetu, zebrana łączna wartość, trafione cele, czas działania). Ponieważ kilka algorytmów jest losowych, uruchomienie więcej niż jednego powtórzenia wypisuje też podsumowanie średnia ± odchylenie standardowe, dzięki czemu widać, jak bardzo wynik danego algorytmu faktycznie różni się między uruchomieniami.

```python
from helpers.benchmark import run_benchmark
run_benchmark(algorithms=["grasp", "tabu_search", "exact_solver"], scenario_files=["../test_data/scenario1.json"])
```

## 4. Jak działa każdy algorytm wyznaczania trasy

Wszystkie 16 algorytmów ma dokładnie taki sam sygnaturę wywołania i kształt wyniku — `(grid, start_row, start_col, max_cost, require_return_to_base=True, blocked_mask=None, ...) → (path, total_value, cost_used)` — dzięki czemu każdy z nich można wymiennie podstawić do `TrajectoryGenerator` lub `run_benchmark`. To, co je różni, to wyłącznie *sposób*, w jaki każdy z nich decyduje, dokąd polecieć dalej.

Dzielą się na cztery rodziny: proste **heurystyki zachłanne**, warianty **przeszukiwania A\***, które optymalnie prowadzą trasę między wybranymi celami, **metaheurystyki**, które iteracyjnie poprawiają całą trasę, oraz jeden **solver dokładny**, który wprost dowodzi optymalności.

**Zakończenie po odnalezieniu wszystkich.** `TrajectoryGenerator.find_best_path` przyjmuje opcjonalny argument `actual_person_locations` — stan faktyczny misji (patrz **Stan faktyczny a wcześniejsze przekonanie** w sekcji 2) — i dokłada dokładnie jedną dodatkową regułę na wierzchu tego, co właśnie zrobił wybrany algorytm, jednolicie, bez modyfikowania logiki żadnego z nich: gdy zwrócona trasa faktycznie przejdzie przez każdą z tych komórek, wszystko po tym punkcie jest odcinane i zastępowane świeżym, krótkim lotem prosto do bazy (jeśli `require_return_to_base`). Działa to dla wszystkich 16 algorytmów bez zmian, ponieważ każdy z nich już teraz zwraca *pełną*, złożoną z pojedynczych komórek trasę, którą dron faktycznie przelatuje, a nie tylko listę punktów pośrednich — więc sprawdzenie to jedynie przeszukanie już obliczonej trasy. Oznacza to też, że osoba jest rozpoznana jako odnaleziona w momencie wejścia na jej komórkę, nawet w połowie lotu w stronę zupełnie innego celu, a nie dopiero gdy algorytm skończy odcinek, na którym akurat się znajdował. Jeśli `actual_person_locations` jest puste lub pominięte, nic się tu nie zmienia — każdy algorytm zachowuje się dokładnie tak, jak opisano poniżej. Lokalizacje nieodnalezione do końca trasy (w tym te pokrywające się z zablokowaną komórką, do której dron nigdy nie może wejść) po prostu oznaczają, że ta reguła nigdy się nie uruchamia, a oryginalny wynik danego algorytmu jest zwracany bez zmian.

Jedna rzecz warta precyzyjnego podkreślenia: dla `exact_solver` nie zmienia to tego, co zostało dowiedzione jako optymalne. Jego optymalizacja w ogóle nie zna pojęcia `actual_person_locations` — wciąż maksymalizuje zebraną wartość ważoną prawdopodobieństwem w ramach budżetu, i ten dowód pozostaje w mocy. Przycięcie zwróconej przez niego trasy, gdy akurat przejdzie ona przez wszystkie prawdziwe osoby, to dokładnie ten sam, kosmetyczny krok wykonywany już po fakcie, zastosowany do wyniku każdego innego algorytmu — nie jest to zmiana samego przeszukiwania ani jego gwarancji.

---

### 4.1 Heurystyki zachłanne (`pathfinding_algorithms/greedy_pathfinding.py`)

Te cztery algorytmy nigdy nie patrzą dalej niż jedna decyzja naprzód i nigdy nie wracają do już podjętego wyboru. Są najszybszymi i najprostszymi algorytmami w projekcie i stanowią łatwy do zrozumienia punkt odniesienia dla wszystkich pozostałych.

#### 4.1.1 `greedy` — sąsiad o najwyższej wartości

**Wyjaśnienie w skrócie:** w każdym kroku dron patrzy na (maksymalnie) 8 komórek bezpośrednio sąsiadujących i przechodzi na tę, która jest warta najwięcej — jak turysta, który na każdym rozwidleniu zawsze wybiera ścieżkę, która z miejsca, w którym stoi, wygląda najlepiej.

**Szczegóły:** z aktualnej komórki każdy z 8 sąsiadów jest oceniany na podstawie swojej surowej wartości (już odwiedzony sąsiad jest oceniany jako „bezwartościowy”, więc ponowne odwiedzenie jest zawsze ostatecznością). Wybierany jest ten dostępny, niezablokowany sąsiad, który ma najwyższą ocenę. Jeśli dwóch lub więcej sąsiadów ma taką samą wartość — co jest bardzo częste, bo większość siatki to nietknięte tło — preferowany jest **tańszy** z wyrównanych ruchów.

**Jak wybiera cele:** w ogóle nie wybiera „celu” w sposób, w jaki robią to inne algorytmy — nie ma żadnego punktu docelowego na myśli, jest tylko powtarzane w kółko porównanie najbliższego otoczenia o zasięgu jednego kroku.

**Kiedy kończy:** w momencie, gdy nie zostaje żaden dostępny, niezablokowany ruch (z uwzględnieniem zarezerwowanego kosztu powrotu do bazy, jeśli jest wymagany) — łącznie z ruchami, które jedynie ponownie odwiedzają już pokryty teren. W przeciwieństwie do każdego kolejnego algorytmu poniżej, nigdy nie kończy wcześniej tylko dlatego, że nie da się dosięgnąć niczego *świeżego*: dopóki istnieje *jakikolwiek* dostępny ruch, nawet czyste powtórzenie starego terenu, kontynuuje, aż budżet naprawdę się wyczerpie. To celowa decyzja projektowa — realny przelot nad niezmapowanym terenem wciąż może mieć sens, nawet tam, gdzie nie ustawiono jeszcze żadnego sygnału o dodatnim prawdopodobieństwie.

**Inne uwagi:** ponieważ nigdy nie patrzy dalej niż na bezpośrednich sąsiadów, nie ma sposobu, by zaplanować ominięcie własnego, już odwiedzonego śladu, i może przez pewien czas błądzić po już pokrytym terenie, zanim znajdzie świeży obszar (albo wcześniej wyczerpie budżet).

**Źródło:** ogólny paradygmat zachłanny „zawsze wybieraj lokalnie najlepszą opcję” to standardowy element projektowania algorytmów — patrz np. T. H. Cormen, C. E. Leiserson, R. L. Rivest, C. Stein, *Introduction to Algorithms*, rozdział „Greedy Algorithms”; a dla jego zastosowania konkretnie do problemów typu orienteering: B. L. Golden, L. Levy, R. Vohra, "The orienteering problem", *Naval Research Logistics*, 1987.

#### 4.1.2 `direct_to_highest_value` — leć prosto do najlepszej komórki, powtarzaj

**Wyjaśnienie w skrócie:** znajdź na całej mapie pojedynczą, najbardziej wartościową nieodwiedzoną komórkę, poleć prosto do niej, potem powtórz — jak planowanie zakupów przystanek po przystanku, zawsze kierując się do najlepszego pozostałego produktu z listy.

**Szczegóły:** cała siatka komórek jest przeszukiwana pod kątem aktualnie najbardziej wartościowej komórki, która wciąż jest nieodwiedzona, niezablokowana i *powyżej progu „prawdziwego celu”* (komórki tła nigdy nie są „gonione”, gdy zabraknie już prawdziwych celów — patrz sekcja 4.6 tego, dlaczego akurat to zabezpieczenie ma znaczenie). Dron dolatuje do niej, korzystając z prostej, szybkiej reguły trasowania: rusza po przekątnej, aż zrówna się na jednej osi, potem leci prosto, obracając się krok po kroku wokół napotkanej przeszkody. Jeśli ten przelot (plus zarezerwowany koszt powrotu) wciąż mieści się w budżecie, jest zatwierdzany, a każda przelatywana komórka — nie tylko cel docelowy — dolicza swoją wartość. Następnie wyszukiwanie powtarza się dla kolejnej, najbardziej wartościowej pozostałej komórki.

**Jak wybiera cele:** zawsze pojedynczy, najbardziej wartościowy pozostały prawdziwy cel na całej siatce, niezależnie od tego, jak daleko się znajduje.

**Kiedy kończy:** w chwili, gdy aktualnie najlepszego celu nie da się w pełni osiągnąć — czy to dlatego, że jest otoczony przeszkodami, których reguła trasowania nie potrafi ominąć, czy dlatego, że przelot plus powrót do bazy nie zmieściłby się w pozostałym budżecie. Nie ma żadnego zapasowego, mniej wartościowego celu; algorytm po prostu się zatrzymuje.

**Inne uwagi:** reguła trasowania (obracanie się wokół jednej przeszkody na raz) to tania heurystyka, a nie pełne przeszukiwanie — może nie znaleźć drogi wokół dużej lub labiryntowej przeszkody, nawet jeśli taka droga istnieje.

**Źródło:** T. Tsiligirides, "Heuristic Methods Applied to Orienteering", *Journal of the Operational Research Society*, 35(9), 1984 — jedna z pierwotnych heurystyk orienteeringowych „zawsze idź do najlepszej pozostałej nagrody”.

#### 4.1.3 `value_cost_ratio` — najlepsza wartość za koszt, w formie turnieju

**Wyjaśnienie w skrócie:** zamiast zawsze gonić pojedynczy, najbogatszy cel bez względu na odległość, ten algorytm pyta „który pobliski cel daje najlepszy zysk *na jednostkę wysiłku*?” — jak wybieranie kolejnej sprawy do załatwienia na podstawie tego, ile załatwia w stosunku do tego, jak bardzo jest „po drodze”, a nie po prostu tego, która sprawa jest największa.

**Szczegóły:** komórki o najwyższej wartości są raz posortowane z góry. W każdej rundzie z tej posortowanej kolejności pobierana jest niewielka liczba jeszcze nieodwiedzonych kandydatów (domyślnie 2), do każdego z nich spekulatywnie wyznaczana jest trasa, a następnie wyliczany jest stosunek **zebrana wartość ÷ koszt przelotu**. Kandydat z najlepszym stosunkiem w danej rundzie jest zatwierdzany; pozostali kandydaci z tej rundy, jeśli wciąż są osiągalni, są zachowywani i ponownie stają do rywalizacji w kolejnej rundzie z nowo dobranym kandydatem — dzięki czemu kandydat nie jest eliminowany tylko dlatego, że raz przegrał.

**Jak wybiera cele:** dowolny osiągalny kandydat — spośród niewielkiej, rotującej puli turniejowej — który aktualnie oferuje najlepszy stosunek zebranej wartości do wydanego kosztu, a nie po prostu najwyższą surową wartość czy najniższy koszt.

**Kiedy kończy:** gdy w danej rundzie żaden kandydat nie jest osiągalny w ramach pozostałego budżetu, bez żadnego zapasowego rozwiązania.

**Inne uwagi:** kandydat, którego nie udało się osiągnąć w danej rundzie, jest trwale odrzucany, a nie ponawiany, ponieważ pozycja drona przesuwa się tylko naprzód, a pozostały budżet tylko maleje — więc nic w kwestii jego osiągalności nie mogłoby się później poprawić.

**Źródło:** B. L. Golden, L. Levy, R. Vohra, "The orienteering problem", *Naval Research Logistics*, 34(3), 1987 — konstrukcja zachłanna oparta na stosunku wartości do kosztu to standardowa heurystyka z tego nurtu badań nad problemem orienteeringowym.

#### 4.1.4 `lowest_cost` — najpierw najtańszy osiągalny cel

**Wyjaśnienie w skrócie:** zawsze leć do tego prawdziwego celu, który jest aktualnie *najtańszy* w dotarciu, zostawiając te drogie na później (albo w ogóle).

**Szczegóły:** taka sama struktura turniejowa jak w `value_cost_ratio` powyżej, ale kandydaci są porównywani wyłącznie na podstawie surowego **kosztu** przelotu, a nie stosunku wartości do kosztu — wygrywa ten osiągalny kandydat w danej rundzie, który jest najtańszy w dotarciu. Szersza pula turniejowa (domyślnie 50 kandydatów, wobec 2 w wersji ze stosunkiem) jest używana właśnie dlatego, że proste porównania samego kosztu wśród wielu, podobnie tanich komórek sąsiadujących z tłem skłaniałyby „spacerowicza” do systematycznego preferowania tej wyrównanej komórki, która akurat była osiągalna po idealnie prostej linii — dając nienaturalną trasę „przyklejoną” do wierszy/kolumn siatki zamiast kierującą się ku obszarom o realnej wartości.

**Jak wybiera cele:** osiągalny kandydat o najniższym koszcie przelotu, kropka — wartość nie odgrywa żadnej roli w samym wyborze (jedynie w tym, które komórki w ogóle liczą się jako kandydaci).

**Kiedy kończy:** gdy w danej rundzie żaden kandydat nie jest osiągalny (zbyt drogi lub fizycznie nieosiągalny) w ramach pozostałego budżetu.

**Inne uwagi:** algorytm ten skłania się ku wykonywaniu na początku misji łatwych, bliskich „zwycięstw” i może zostawić dużo niewykorzystanego budżetu, jeśli pozostałe nieodwiedzone cele są relatywnie drogie — nigdy nie waży „czy to warte swojego kosztu”, tylko „czy to tanie”.

**Źródło:** ta sama rodzina heurystyk problemu orienteeringowego co w 4.1.2/4.1.3 (Golden, Levy i Vohra, 1987); ten wariant odpowiada czystej regule konstrukcji w stylu „najbliższy sąsiad”.

---

### 4.2 Warianty przeszukiwania A* (`pathfinding_algorithms/astar_pathfinding.py`)

Cztery zachłanne algorytmy powyżej wyznaczają trasę między komórkami za pomocą taniej, przybliżonej heurystyki „obracaj się wokół jednej przeszkody na raz” — szybkiej, lecz niegwarantującej znalezienia rzeczywiście najtańszej trasy i podatnej na utknięcie przy złożonym układzie przeszkód. Cztery algorytmy z tej rodziny zamieniają tę heurystykę na prawdziwe **[przeszukiwanie A\*](https://en.wikipedia.org/wiki/A*_search_algorithm)**: realne przeszukiwanie grafu, które *gwarantuje* znalezienie najtańszej możliwej trasy między dowolnymi dwiema komórkami (nigdy nie przeszacowując prawdziwego pozostałego kosztu dzięki *heurystyce dopuszczalnej* — odległości w linii prostej pomnożonej przez najtańszy możliwy koszt pojedynczego kroku gdziekolwiek na mapie), kosztem przeszukania większej części siatki, by tę trasę znaleźć.

Trzy z czterech (`a_star_to_highest_value`, `a_star_value_cost_ratio`, `a_star_lowest_cost`) mają poza tym identyczną logikę wyboru celów co ich zachłanne odpowiedniki z sekcji 4.1 — zmienia się jedynie sposób trasowania między wybranymi punktami.

#### 4.2.1 `a_star_to_highest_value`

Taka sama logika wyboru celów jak w `direct_to_highest_value` (4.1.2): wielokrotnie szuka pojedynczego, najbardziej wartościowego pozostałego prawdziwego celu na siatce. Jedyna różnica to *sposób* dotarcia do niego — przez prawdziwe przeszukiwanie A* zamiast heurystyki „obracaj się wokół przeszkody” — dzięki czemu potrafi nawigować przez duży lub labiryntowy `blocked_mask`, z którym prostsza heurystyka mogłaby sobie nie poradzić, i ma gwarancję znalezienia rzeczywiście najtańszej trasy, gdy istnieje więcej niż jedna możliwa. Kończy działanie na tych samych warunkach co 4.1.2: w chwili, gdy aktualnie najlepszego celu nie da się w pełni, w ramach budżetu osiągnąć, bez zapasowego, mniej wartościowego celu.

#### 4.2.2 `a_star_value_cost_ratio`

Taka sama logika turniejowa jak w `value_cost_ratio` (4.1.3) — niewielka, rotująca pula kandydatów, wygrywa najlepszy stosunek zebranej wartości do kosztu w danej rundzie — ale koszt każdego kandydata to prawdziwy, optymalny koszt wyznaczony przez A*, a nie oszacowanie tańszej heurystyki. Kończy działanie tak samo: gdy w danej rundzie żaden kandydat nie jest osiągalny w ramach budżetu.

#### 4.2.3 `a_star_lowest_cost`

Taka sama logika turniejowa jak w `lowest_cost` (4.1.4) — szeroka, rotująca pula (domyślnie 50) — wygrywa najtańszy osiągalny kandydat w danej rundzie — z prawdziwymi, optymalnymi kosztami A*. Kończy działanie tak samo.

#### 4.2.4 `namoa_star` — wieloobiektywowe A*

**Wyjaśnienie w skrócie:** wyobraź sobie planowanie spaceru, w którym zależy ci jednocześnie na *dwóch* rzeczach — jak bardzo się zmęczysz i ile naklejek uda ci się zebrać po drodze z automatów rozsianych po mieście. Zwykły planista najkrótszej trasy minimalizuje wyłącznie zmęczenie i nigdy nie rozważy nieco dłuższej trasy, która zbiera pięć naklejek niemal bez dodatkowego wysiłku — w ogóle nie myśli o naklejkach. **[NAMOA\*](https://en.wikipedia.org/wiki/Multi-objective_optimization)** („New Approach to Multi-Objective A\*”, nowe podejście do wieloobiektywowego A*) jest mądrzejsze: zamiast znajdować *jedną* trasę, wyznacza całe menu naprawdę dobrych kompromisów — dla każdej sensownej liczby „ile naklejek”, najmniej męczącą trasę, która ją osiąga — a następnie wybiera z tego menu pozycję najlepiej pasującą do celu.

**Szczegóły:** wykorzystuje taką samą strukturę turniejową jak `a_star_value_cost_ratio` (4.2.2), ale zamiast wyznaczać tylko pojedynczą *najtańszą* trasę do kandydata (jak zwykłe A*), przeprowadza pełne przeszukiwanie wieloobiektywowe, śledzące każdy **niezdominowany** kompromis (koszt, zebrana wartość) prowadzący do tego kandydata — trasa jest zachowywana tylko wtedy, gdy żadna inna trasa do tego samego punktu nie jest jednocześnie nie droższa *i* nie mniej wartościowa. Spośród całego tego menu kompromisów dla danego kandydata wybierany jest punkt o najlepszym stosunku wartości do kosztu, który służy do oceny kandydata w danej rundzie. Ponieważ zwykłe A* wyznaczające najkrótszą trasę nigdy nawet nie rozważy droższego objazdu, który przy okazji zbiera dodatkową wartość po drodze, NAMOA* może czasem znaleźć odpowiedź, która jest naprawdę lepsza z punktu widzenia właściwego celu tego projektu (zebrać jak najwięcej wartości w ramach budżetu), niż potrafi to zrobić samo A* wyznaczające najtańszą trasę.

**Jak wybiera cele:** ten sam mały, rotujący turniej co w 4.2.2, oceniany na podstawie najlepszego dostępnego punktu wartość/koszt z pełnego menu kompromisów Pareto-optymalnych każdego kandydata, a nie pojedynczej liczby reprezentującej najtańszą trasę.

**Kiedy kończy:** gdy w danej rundzie żaden kandydat nie ma już dostępnego, osiągalnego punktu kompromisu — ta sama zasada zakończenia co w pozostałych algorytmach turniejowych.

**Inne uwagi:** śledzenie całego zbioru kompromisów na komórkę, zamiast jednej najlepszej liczby, sprawia, że jest to zdecydowanie najbardziej kosztowny obliczeniowo algorytm w projekcie — na dużym, rozległym scenariuszu pojedyncza ocena kandydata może zająć sekundy, nawet przy agresywnym odcinaniu wyraźnie gorszych tras cząstkowych. To właściwe narzędzie, gdy sam kompromis wartość/koszt jest istotą sprawy, albo gdy przestrzeń przeszukiwania jest na tyle mała, by dało się ją dokładnie przeszukać; dla dużych scenariuszy z wieloma rozproszonymi celami `a_star_value_cost_ratio` lub zwykłe `value_cost_ratio` dają niezawodnie szybsze, wciąż mocne wyniki.

**Źródło:** L. Mandow, J. L. Pérez-de-la-Cruz, "A New Approach to Multiobjective A\* Search", *Proceedings of IJCAI 2005*; w oparciu o oryginalną pracę o A*: P. E. Hart, N. J. Nilsson, B. Raphael, "A Formal Basis for the Heuristic Determination of Minimum Cost Paths", *IEEE Transactions on Systems Science and Cybernetics*, 4(2), 1968.

---

### 4.3 Metaheurystyki (`pathfinding_algorithms/metaheuristic_pathfinding.py`)

Każdy algorytm opisany dotychczas podejmuje decyzję i nigdy do niej nie wraca. Siedem algorytmów z tej rodziny działa inaczej — operuje na **całej kandydackiej trasie naraz** — uporządkowanej liście prawdziwych celów do odwiedzenia — i wielokrotnie próbuje ją *poprawić*, czasem nawet celowo akceptując chwilowo gorszą trasę, właśnie po to, by móc uciec od decyzji, która lokalnie wyglądała nie do pobicia, choć w rzeczywistości nie była najlepszą dostępną. Sześć z siedmiu (wszystkie poza optymalizacją mrówkową) współdzieli dokładnie tę samą reprezentację rozwiązania i dokładnie te same trzy podstawowe ruchy — **ADD** (dodaj nieodwiedzony cel), **DROP** (usuń odwiedzony) i **SWAP** (zamień kolejność dwóch) — a trasa na poziomie siatki między wybranymi celami jest zawsze wyznaczana w ten sam sposób (to samo trasowanie „obracaj się wokół przeszkody”, którego używają algorytmy zachłanne). To, co je różni, to wyłącznie *strategia* przeszukiwania tej przestrzeni możliwych tras.

Ponieważ większość z nich korzysta z losowości, uruchomienie ich więcej niż raz na tym samym scenariuszu (np. przez parametr `repetitions` w `run_benchmark`) na ogół da nieco inne — choć zwykle podobnie dobre — wyniki za każdym razem.

#### 4.3.1 `ant_colony` — optymalizacja mrówkowa (ang. Ant Colony Optimization, ACO)

**Wyjaśnienie w skrócie:** wzorowana na tym, jak prawdziwe kolonie mrówek znajdują krótkie trasy między gniazdem a pożywieniem. Każda mrówka wędruje w miarę losowo, zostawiając po drodze ślad zapachowy (feromon). Zapach z czasem blaknie, ale krótsza lub lepsza trasa pozwala większej liczbie mrówek pokonać ją w obie strony w tym samym czasie, więc gromadzi zapach szybciej niż gorsza trasa — co czyni ją bardziej atrakcyjną dla kolejnych mrówek, co dodatkowo ją wzmacnia. Żadna pojedyncza mrówka nigdy nie widzi wiedzy całej kolonii; to właśnie wspólny, blaknący ślad pozwala kolonii wspólnie zbiec do dobrej trasy.

**Szczegóły:** przez ustaloną liczbę iteracji każda z kilkunastu mrówek niezależnie buduje jedną kompletną kandydacką trasę krok po kroku, zaczynając od bazy. W każdym kroku, stojąc w danej komórce, kolejna komórka jest wybierana **losowo spośród wszystkich możliwych sąsiadów**, ale z wagą `(feromon na tej krawędzi)^alfa × (atrakcyjność celu ÷ koszt ruchu)^beta` — więc krok jest bardziej prawdopodobny, jeśli krawędź ma już silny ślad feromonowy albo sama w sobie wygląda obiecująco. Ponieważ krok-po-kroku spacer może „widzieć” jedynie swoich bezpośrednich 8 sąsiadów, z góry raz wyznaczane jest **pole atrakcyjności**: wartość każdego prawdziwego celu promieniuje na zewnątrz z wykładniczym zanikiem wraz z odległością, więc komórka oddalona o kilka kroków od wartościowego celu wciąż wygląda lokalnie atrakcyjnie na długo przed tym, jak mrówka faktycznie stanie tuż obok niego — bez tego rzadko rozmieszczona mapa celów byłaby niemal niewidoczna dla czysto lokalnej, jednokrokowej reguły decyzyjnej. Gdy wszystkie mrówki w danej rundzie skończą, cały feromon wyparowuje o ustaloną część, po czym każda mrówka nakłada świeży feromon na krawędzie, których faktycznie użyła, proporcjonalnie do tego, jak dobra (wartość zebrana na jednostkę kosztu) była jej ukończona trasa.

**Jak wybiera cele:** nie „wybiera celów” wprost, tak jak robią to algorytmy oparte na trasie poniżej — każda mrówka po prostu idzie krok po kroku, kierowana kombinacją feromonu i pola atrakcyjności, naturalnie ciążąc ku wartościowym obszarom mapy, nigdy wcześniej nie zobowiązując się do konkretnego celu.

**Kiedy kończy:** spacer pojedynczej mrówki kończy się, gdy nie ma już żadnego możliwego sąsiada (patrz *inne uwagi* — mechanizm zapasowy na wypadek, gdy zdarzy się to przedwcześnie); całe przeszukiwanie kończy się po ustalonej liczbie iteracji i zwraca pojedynczą najlepszą trasę, jaką znalazła którakolwiek mrówka w całym przebiegu.

**Inne uwagi:** ponieważ mrówka zawsze wchodzi tylko na sąsiednią, jeszcze nieodwiedzoną komórkę, może w zasadzie uwięzić samą siebie wewnątrz własnego, już pokrytego śladu na długo przed wyczerpaniem budżetu. Gdy to się zdarzy, uruchamia się mechanizm naprawczy: mrówka patrzy na „front” (nieodwiedzone komórki graniczące z już pokrytym terenem), próbuje dolecieć do kilku najbardziej atrakcyjnych komórek frontu i wznawia trasę od tej, która okaże się osiągalna — poddając się całkowicie tylko wtedy, gdy żadna z tych prób też się nie powiedzie.

**Źródło:** M. Dorigo, *Optimization, Learning and Natural Algorithms*, praca doktorska, Politecnico di Milano, 1992; M. Dorigo, L. M. Gambardella, "Ant Colony System: A Cooperative Learning Approach to the Traveling Salesman Problem", *IEEE Transactions on Evolutionary Computation*, 1(1), 1997.

#### 4.3.2 `tabu_search` — przeszukiwanie tabu (Tabu Search)

**Wyjaśnienie w skrócie:** śledzi **pojedynczą**, ewoluującą kandydacką trasę i wielokrotnie pyta „jaka jest najlepsza pobliska zmiana, jaką mogę wprowadzić do tego, co już mam?”. Jego charakterystyczną sztuczką jest krótkotrwała pamięć niedawno wykonanych ruchów, których odmawia natychmiast cofnąć — to właśnie pozwala mu czasem celowo zaakceptować *gorszy* ruch, by wydostać się ze ślepego zaułka, w którym przeszukiwanie idące wyłącznie „pod górę” utknęłoby na zawsze.

**Szczegóły:** w każdej rundzie generowane są wszystkie ruchy ADD/DROP/SWAP osiągalne z aktualnej trasy, te niedopuszczalne (przekraczające budżet) są odrzucane, a wykonywany jest ten *dopuszczalny, nieobjęty zakazem* ruch, który daje najlepszy wynik w danej rundzie — nawet jeśli oznacza to krok wstecz pod względem wartości. Ruch, który właśnie wykonano, ma swoje odwrócenie oznaczone jako „tabu” (zakazane) na ustaloną liczbę rund, właśnie po to, by przeszukiwanie nie mogło natychmiast pożałować i cofnąć własnej ostatniej decyzji — chyba że to cofnięcie faktycznie pobiłoby najlepszą trasę widzianą w całym dotychczasowym przebiegu, w którym to wypadku jest mimo wszystko dozwolone (kryterium aspiracji: wystarczająco dobry wynik może złamać regułę).

**Jak wybiera cele:** spośród wszystkich osiągalnych sąsiadów ADD/DROP/SWAP aktualnej trasy — ten, który daje najlepszy wynik w danej rundzie, z zastrzeżeniem listy tabu.

**Kiedy kończy:** po ustalonej liczbie rund albo natychmiast, jeśli w danej rundzie każdy możliwy ruch jest albo niedopuszczalny, albo zablokowany przez listę tabu — w zależności od tego, co nastąpi pierwsze. Zwracana jest pojedyncza najlepsza trasa widziana w *dowolnym* momencie przebiegu, niekoniecznie ta, przy której przeszukiwanie akurat się zatrzymało (mogło od tamtej pory zawędrować w gorsze miejsce).

**Inne uwagi:** ocena każdego możliwego ruchu oznacza ponowne przejście całej trasy od zera za każdym razem — w porządku dla dziesiątek do niższych setek prawdziwych celów typowego scenariusza, ale nie skaluje się dobrze do siatki z tysiącami indywidualnie wartościowych komórek.

**Źródło:** F. Glover, "Future Paths for Integer Programming and Links to Artificial Intelligence", *Computers & Operations Research*, 13(5), 1986; F. Glover, M. Laguna, *Tabu Search*, Kluwer Academic Publishers, 1997.

#### 4.3.3 `variable_neighborhood_search` — przeszukiwanie zmiennego sąsiedztwa (VNS)

**Wyjaśnienie w skrócie:** wyobraź sobie stanie na pagórkowatym krajobrazie „jak dobra jest ta trasa”, próbując znaleźć najwyższe wzgórze. Zwykły algorytm wspinaczkowy robi jeden malutki krok naraz w stronę tego, co wygląda lepiej właśnie teraz — co działa świetnie, dopóki nie stanie na szczycie *jakiegoś* wzgórza, niekoniecznie *najwyższego* w okolicy, bez żadnego małego kroku, który wyglądałby na lepszy. VNS radzi sobie z tym tak: gdy małe kroki przestają pomagać, wykonuje zamiast tego większy, **losowy** skok — na tyle daleki, że może wylądować w zupełnie innym miejscu — a potem znów wraca do małych, ostrożnych kroków tam, gdzie wylądował. Jeśli to nie pomogło, następnym razem wykonuje jeszcze większy skok; w chwili, gdy skok faktycznie się opłaci, wraca do małych skoków.

**Szczegóły:** każda runda ma dwie fazy. **Wstrząśnięcie (shake):** losowo usuwa `k` celów z aktualnej trasy i dodaje `k` losowych, jeszcze nieuwzględnionych — destrukcyjny, losowy skok, którego rozmiar `k` zaczyna się od małej wartości. **Przeszukiwanie lokalne:** od tak wstrząśniętej trasy wielokrotnie wykonywany jest pojedynczy ruch ADD/DROP/SWAP, który najbardziej ją poprawia, aż nic więcej nie pomaga (lokalny szczyt). Jeśli powstała trasa jest lepsza niż ta, od której runda się zaczęła, zostaje zachowana, a `k` wraca do najmniejszej wartości; jeśli nie, trasa pozostaje bez zmian, ale `k` rośnie, więc skok w kolejnej rundzie będzie większy.

**Jak wybiera cele:** faza wstrząśnięcia wybiera losowo (bez żadnej oceny); następująca po każdym wstrząśnięciu faza przeszukiwania lokalnego wybiera pojedynczy, najlepiej oceniony dostępny ruch ADD/DROP/SWAP, dokładnie tak jak jedna runda wyboru ruchu w przeszukiwaniu tabu (bez pamięci tabu).

**Kiedy kończy:** po ustalonej liczbie rund; zwracana jest najlepsza trasa widziana w dowolnym momencie.

**Inne uwagi:** kluczowa różnica względem przeszukiwania tabu polega na tym, *jak* każdy z nich ucieka ze ślepego zaułka — przeszukiwanie tabu pozostaje zdyscyplinowane, wykonując jeden ostrożny krok i pamięć „bez cofania się”; VNS natomiast fizycznie przenosi się poprzez czasem duży, losowy skok, a potem porządkuje sytuację lokalnie tam, gdzie wylądował. Wstrząśnięta trasa może na starcie przekraczać budżet; następująca po niej faza przeszukiwania lokalnego zwykle sama to naprawia (najczęściej ruchem DROP).

**Źródło:** N. Mladenović, P. Hansen, "Variable Neighborhood Search", *Computers & Operations Research*, 24(11), 1997.

#### 4.3.4 `grasp` — Greedy Randomized Adaptive Search Procedure (GRASP)

**Wyjaśnienie w skrócie:** wyobraź sobie budowanie wieży element po elemencie. Czysto zachłanny budowniczy zawsze chwyta pojedynczy, najlepszy dostępny element — ale to zawsze utrwala dokładnie te same wybory, a wczesny, „najlepiej wyglądający” element nie zawsze prowadzi do najlepszej ukończonej wieży. GRASP radzi sobie z tym tak: na każdym kroku patrzy na *garść* najlepszych dostępnych elementów, nie tylko na pojedynczy najlepszy, i losowo wybiera jeden z nich. Wciąż buduje to naprawdę dobrą wieżę, ale niemal za każdym razem inną. Po zbudowaniu — dopracowuje ją, a potem odrzuca cały plan i buduje od zera zupełnie nową wieżę w ten sam sposób — wielokrotnie — zachowując tę ukończoną próbę, która okazała się ostatecznie najlepsza.

**Szczegóły:** każda z ustalonej liczby niezależnych iteracji ma dwie fazy. **Konstrukcja:** zaczynając od końca aktualnej trasy, każdy osiągalny, pozostały cel jest oceniany stosunkiem zebrana wartość/koszt, kilka najlepszych („ograniczona lista kandydatów”, RCL) jest zachowywanych, a jeden z nich jest wybierany *losowo z równym prawdopodobieństwem* — powtarzane, z ponowną oceną za każdym razem od nowa (część „adaptacyjna”), aż nic więcej nie będzie osiągalne. **Przeszukiwanie lokalne:** dokładnie ta sama wspinaczka, której VNS używa po wstrząśnięciu, zastosowana do świeżo skonstruowanej trasy. Zachowywany jest wynik tej iteracji, która okazała się ogólnie najlepsza.

**Jak wybiera cele:** podczas konstrukcji — jeden z najlepszych `rcl_size` osiągalnych celów według stosunku wartość/koszt, wybrany losowo, a nie zawsze pojedynczy najlepszy; podczas przeszukiwania lokalnego — dowolny pojedynczy ruch ADD/DROP/SWAP o najlepszym wyniku (tak samo jak w przeszukiwaniu tabu/VNS).

**Kiedy kończy:** po ustalonej liczbie niezależnych powtórzeń konstrukcja+przeszukiwanie lokalne; zwracany jest najlepszy wynik spośród wszystkich.

**Inne uwagi:** w przeciwieństwie do przeszukiwania tabu, VNS czy symulowanego wyżarzania, GRASP **nie przenosi żadnej pamięci** między iteracjami — każde powtórzenie jest w pełni niezależne, co czyni go naturalnie równoległym i odpornym na uwięzienie przez złą wczesną decyzję odziedziczoną po poprzedniej próbie, kosztem tego, że nigdy nie buduje na częściowym postępie tak, jak robią to pozostałe algorytmy.

**Źródło:** T. A. Feo, M. G. C. Resende, "Greedy Randomized Adaptive Search Procedures", *Journal of Global Optimization*, 6(2), 1995.

#### 4.3.5 `simulated_annealing` — symulowane wyżarzanie (SA)

**Wyjaśnienie w skrócie:** wyobraź sobie odbijającą się piłeczkę upuszczoną na ten sam pagórkowaty krajobraz „jak dobra jest ta trasa”. Piłeczka, która porusza się wyłącznie w stronę czegoś lepszego, utyka na pierwszym małym wzgórzu, na jakie się wespnie. Symulowane wyżarzanie sprawia, że piłeczka na starcie odbija się mocniej — dzięki czemu potrafi przeskakiwać przez małe wzgórza i płytkie doliny, wędrując dość swobodnie, czasem nawet w wyraźnie gorsze miejsce. Sprężystość jest następnie bardzo stopniowo obniżana („chłodzenie”), tak by piłeczka osiadła, miejmy nadzieję, na wysokim wzgórzu, na które natrafiła, gdy miała jeszcze dużo energii do eksploracji. Nazwa pochodzi od wyżarzania metalu: podgrzej go, a potem chłodź *powoli*, by jego atomy miały czas ułożyć się w dobrą strukturę; schłodź zbyt szybko, a zastygnie w złej.

**Szczegóły:** każda iteracja proponuje dokładnie **jeden** losowy ruch ADD/DROP/SWAP. Jeśli jest lepszy, zawsze zostaje zaakceptowany. Jeśli jest gorszy, wciąż jest akceptowany z prawdopodobieństwem, które maleje zarówno wraz z tym, jak bardzo ruch jest gorszy, jak i wraz z chłodzeniem „temperatury” w trakcie przebiegu (`prawdopodobieństwo = exp(o ile gorzej ÷ temperatura)`) — kryterium Metropolisa. Temperatura zaczyna wysoko (swobodna eksploracja) i jest mnożona przez stały współczynnik chłodzenia w każdej rundzie, nigdy nie spadając poniżej niewielkiego minimum.

**Jak wybiera cele:** w każdej rundzie proponowany jest pojedynczy, losowy ruch ADD, DROP lub SWAP — nieoceniany na tle alternatyw tak, jak przeszukiwanie tabu/VNS/GRASP porównują całą partię naraz, po prostu akceptowany lub odrzucany na podstawie własnych zalet (i temperatury).

**Kiedy kończy:** po ustalonej liczbie iteracji — zwykle dużo większej niż w przeszukiwaniu tabu czy VNS, ponieważ pojedyncza runda jest tu dużo tańsza (jedna propozycja, nie całe sąsiedztwo). Zwracana jest najlepsza trasa widziana w dowolnym momencie.

**Inne uwagi:** temperatura nigdy nie spada dokładnie do zera, zarówno by uniknąć dzielenia przez zero, jak i dlatego, że temperatura dokładnie równa zero uczyniłaby przeszukiwanie czysto zachłannym do końca przebiegu — niewielkie minimum utrzymuje odrobinę losowości przez cały czas.

**Źródło:** S. Kirkpatrick, C. D. Gelatt, M. P. Vecchi, "Optimization by Simulated Annealing", *Science*, 220(4598), 1983.

#### 4.3.6 `genetic_algorithm` — algorytm genetyczny (GA)

**Wyjaśnienie w skrócie:** wyobraź sobie hodowanie roślin dla jak najwyższego, najbardziej wydajnego ogrodu. Nie wybrałbyś po prostu pojedynczej najlepszej rośliny i nie zatrzymał się na tym — pozwoliłbyś swoim najlepszym roślinom się krzyżować, łącząc ich cechy, dopuściłbyś kilka losowych mutacji i wyhodował nowe pokolenie z tego wszystkiego. Powtarzaj to przez wiele pokoleń, a ogród jako całość ma tendencję do stałej poprawy, ponieważ dobre cechy od dwóch *różnych* rodziców mogą połączyć się w potomka lepszego niż którykolwiek z rodziców z osobna.

**Szczegóły:** naraz utrzymywana jest cała populacja kandydackich tras („chromosomów”) — część zasiana szybką, losowo-zachłanną konstrukcją (tą samą, której używa GRASP), reszta to w pełni losowe podzbiory. Każde pokolenie: każdy chromosom jest oceniany na podstawie zebranej wartości (chromosom zbyt ambitny jak na budżet jest „naprawiany” — przechodzony cel po celu i po prostu ucinany na pierwszym celu, który się nie mieści, zamiast być całkowicie odrzucanym, tak by zbyt gorliwy wynik krzyżowania nie szedł na marne). Rodzice są wybierani przez selekcję turniejową (kilka losowych chromosomów, najlepiej dopasowany wygrywa szansę na rozmnażanie). Dwoje rodziców tworzy potomka przez krzyżowanie — potomek zachowuje prefiks planu jednego rodzica o losowej długości i uzupełnia resztę tymi celami drugiego rodzica, których jeszcze nie ma. Potomek jest czasem mutowany dokładnie tym samym pojedynczym, losowym ruchem ADD/DROP/SWAP, którego używa symulowane wyżarzanie. Najlepsze chromosomy („elity”) zawsze przechodzą do kolejnego pokolenia bez zmian, dzięki czemu najlepszy osobnik w populacji nigdy nie może przypadkiem się pogorszyć z jednego pokolenia na drugie.

**Jak wybiera cele:** pośrednio, poprzez cały proces populacja + selekcja + krzyżowanie + mutacja, a nie przez jakikolwiek pojedynczy, jawny krok „wybierz najlepszego kandydata” — dobre kombinacje celów, które pojawiają się u dopasowanych rodziców, są przekazywane dalej i rekombinowane.

**Kiedy kończy:** po ustalonej liczbie pokoleń; zwracany jest pojedynczy, najlepszy chromosom widziany kiedykolwiek (niekoniecznie w ostatnim pokoleniu).

**Inne uwagi:** każda metaheurystyka w tym projekcie współdzieląca reprezentację trasy jako „uporządkowana lista celów” — przeszukiwanie tabu, VNS, GRASP, symulowane wyżarzanie i ten algorytm — ostatecznie eksploruje te same trzy podstawowe ruchy (ADD/DROP/SWAP); to, co je wszystkie różni, to wyłącznie *strategia* wyboru spośród tych ruchów.

**Źródło:** J. H. Holland, *Adaptation in Natural and Artificial Systems*, University of Michigan Press, 1975; D. E. Goldberg, *Genetic Algorithms in Search, Optimization, and Machine Learning*, Addison-Wesley, 1989.

#### 4.3.7 `large_neighborhood_search` — adaptacyjne przeszukiwanie dużego sąsiedztwa (ALNS)

**Wyjaśnienie w skrócie:** wyobraź sobie już zbudowany, niezły zamek z klocków, w którym kilka pomieszczeń dałoby się prawdopodobnie przestawić, by zmieścić więcej wież. Zamiast burzyć całość i zaczynać od nowa (podejście GRASP) albo zmieniać jedno pomieszczenie na raz (podejście przeszukiwania tabu/VNS), ALNS wyburza *garść* pomieszczeń naraz — zostawiając większość dobrego zamku na miejscu — i odbudowuje tylko brakującą część tak sprytnie, jak potrafi. W trakcie przebiegu wypróbowywanych jest kilka różnych sposobów „wyburzania” i „odbudowywania” fragmentu, a te, które się sprawdzają, są używane coraz częściej, w miarę jak przeszukiwanie uczy się, co faktycznie działa na tej konkretnej mapie.

**Szczegóły:** zaczyna od realnej, już niezłej trasy (zbudowanej tą samą losowo-zachłanną metodą, którą GRASP konstruuje swoją). W każdej rundzie: **operator wyburzający** usuwa fragment aktualnej trasy (losowy podzbiór; przystanki wnoszące najmniej wartości; albo klaster *geograficznie powiązanych* przystanków, ponieważ bliskie sobie cele zwykle konkurują o tę samą decyzję „czy tu wpaść”, więc usunięcie ich razem, jako grupy, daje realną swobodę przetasowania). **Operator naprawiający** następnie dodaje przystanki z powrotem — czerpiąc z *wszystkich* prawdziwych celów aktualnie nieujętych w trasie, nie tylko tych właśnie usuniętych, ponieważ zwolniony budżet może teraz pozwolić na coś lepszego — albo zawsze wstawiając parę kandydat/pozycja o najlepszym wyniku, albo priorytetyzując kandydatów, którzy mają tylko jedno dobre pozostałe miejsce do wstawienia (i straciliby tę szansę, gdyby nie zostali wzięci teraz). Naprawiona trasa jest akceptowana, gdy jest lepsza, a czasem akceptowana nawet, gdy jest gorsza (to samo kryterium Metropolisa co w symulowanym wyżarzaniu, z własną chłodzącą się temperaturą). Ostatnie osiągnięcia każdego operatora wyburzającego/naprawiającego są oceniane punktowo, a co jakiś czas szanse na ponowny wybór danego operatora są przesuwane w stronę tych, które faktycznie się sprawdzają.

**Jak wybiera cele:** operatory wyburzające usuwają fragment celów z *aktualnej* trasy (losowo, według najmniejszego wkładu albo według geograficznego skupienia); operatory naprawiające dodają z powrotem cele z *całej* puli prawdziwych celów aktualnie nieuwzględnionych w trasie, priorytetyzując po najlepszym wyniku albo po pilności typu „teraz albo nigdy”.

**Kiedy kończy:** po ustalonej liczbie rund; zwracana jest najlepsza trasa widziana w dowolnym momencie.

**Inne uwagi:** w przeciwieństwie do pojedynczych ruchów ADD/DROP/SWAP w przeszukiwaniu tabu/VNS, ten algorytm potrafi naprawdę odkryć poprawę wymagającą jednoczesnej zmiany *kilku* celów, ponieważ krok wyburzania usuwa je jako grupę, zanim krok naprawiający w ogóle musi wybrać kombinację zastępczą — przetasowanie, które na każdym pojedynczym, pośrednim kroku „jeden ruch na raz” wyglądałoby gorzej, i którego algorytm zmieniający tylko jedną rzecz naraz nigdy by nie odkrył, tutaj można znaleźć bezpośrednio.

**Źródło:** P. Shaw, "Using Constraint Programming and Local Search Methods to Solve Vehicle Routing Problems", *Proceedings of CP 1998*; S. Ropke, D. Pisinger, "An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows", *Transportation Science*, 40(4), 2006.

---

### 4.4 Solver dokładny (`pathfinding_algorithms/exact_pathfinding.py`)

#### 4.4.1 `exact_solver` — rozwiązanie dowiedzione jako optymalne, przez CP-SAT

**Wyjaśnienie w skrócie:** każdy algorytm powyżej szuka *dobrej* odpowiedzi, nigdy nie mogąc dowieść, że jest to odpowiedź *najlepsza z możliwych*. Ten algorytm różni się rodzajowo: przekazuje problem prawdziwemu silnikowi optymalizacji matematycznej — [solverowi CP-SAT z Google OR-Tools](https://developers.google.com/optimization) — który przeszukuje przestrzeń możliwych odpowiedzi w sposób ustrukturyzowany, pozwalający *dowieść*, kiedy znalazł prawdziwe optimum, albo, jeśli wcześniej zabraknie mu czasu, dokładnie zaraportować, jak daleko jego najlepsza odpowiedź może wciąż być od optimum.

**Szczegóły:** siatka jest najpierw redukowana do samej komórki startowej plus każdego prawdziwego celu (tego samego zbioru celów, którego używa każda metaheurystyka z sekcji 4.3), a prawdziwy, najtańszy koszt przelotu między każdą parą z nich jest wyliczany z góry za pomocą prawdziwego przeszukiwania A* (sekcja 4.2). Zamienia to „znajdź dobrą trasę na siatce” w znacznie mniejszy, klasyczny **problem orienteeringowy** ([Orienteering Problem](https://en.wikipedia.org/wiki/Orienteering_problem)): wybierz podzbiór tych celów oraz kolejność ich odwiedzania. Ten problem jest następnie modelowany jako przeszukiwanie metodą **[podziału i ograniczeń](https://en.wikipedia.org/wiki/Branch_and_bound)** (ang. *branch-and-bound*): każdy cel otrzymuje opcję „pomiń mnie”, każda para celów otrzymuje opcję „leć bezpośrednio między nimi”, a pojedyncze ograniczenie (`AddCircuit` z CP-SAT) wymusza, by cele, których nie pominięto, tworzyły jedną spójną trasę, zaczynającą i kończącą się w bazie. Solver wielokrotnie relaksuje problem, by szybko uzyskać górne ograniczenie, i tylko wtedy zagłębia się dalej w daną gałąź możliwości, gdy to ograniczenie wskazuje, że wciąż mogłaby ona pobić najlepszą odpowiedź znalezioną dotychczas gdziekolwiek indziej w przeszukiwaniu — wszystko inne jest matematycznie wykluczane bez potrzeby sprawdzania. To właśnie pozwala mu ostatecznie *dowieść* optymalności, zamiast jedynie zaraportować „to najlepsze, co udało mi się znaleźć”.

**Jak wybiera cele:** w ogóle nie wybiera przyrostowo — solver rozważa każdy możliwy podzbiór i kolejność puli prawdziwych celów naraz (poprzez strukturę ograniczeń, a nie brutalną siłę) i zwraca ten rzeczywiście najlepszy w ramach ograniczenia budżetowego.

**Kiedy kończy:** albo w chwili, gdy udowodni, że nie może istnieć lepsza odpowiedź („OPTIMAL”), albo gdy wcześniej upłynie konfigurowalny limit czasu — w tym wypadku wciąż zwraca swoją najlepszą dotychczas znalezioną odpowiedź, razem z wypisanym na konsolę luką optymalności — gwarantowanym ograniczeniem tego, jak daleko od prawdziwego optimum ta odpowiedź może jeszcze być. Kontrakt zwracanej wartości w tym projekcie to ta sama, prosta krotka `(path, total_value, cost_used)`, której używa każdy inny algorytm, więc dodatkowe informacje diagnostyczne (status rozwiązania, wartość celu, najlepsze pozostałe ograniczenie, luka) są wypisywane na konsolę, a nie zwracane.

**Inne uwagi:** to celowo **nie jest** ręcznie napisany algorytm przeszukiwania — faktyczny silnik podziału i ograniczeń to gotowy, produkcyjny solver CP-SAT od Google, a nie kod napisany specjalnie dla tego projektu; specyficzne dla projektu jest jedynie przełożenie „siatki i budżetu” na zmienne i ograniczenia zrozumiałe dla CP-SAT. Dwa koszty sprawiają, że jest to niepraktyczne na bardzo dużych scenariuszach: wyliczenie z góry każdego parowego kosztu przelotu jest kwadratowe względem liczby prawdziwych celów, a sam leżący u podstaw problem orienteeringowy jest NP-trudny — dlatego zabezpieczenie (`max_targets`, domyślnie 60) odmawia nawet podjęcia próby na scenariuszu z większą liczbą prawdziwych celów, chyba że zostanie to jawnie wymuszone. Warto też precyzyjnie określić, co dokładnie oznacza tu „optymalny”: funkcja celu dolicza wartość danego celu tylko wtedy, gdy jest on jawnie wybrany jako przystanek — a nie żadną *przygodną* wartość, którą wybrana trasa akurat mija po drodze gdzie indziej, tak jak czasem okazyjnie robi to tańsza heurystyka trasowania metaheurystyk opartych na trasie. Solver dowodzi więc najlepszego możliwego wyboru *którzy cele odwiedzić i w jakiej kolejności*, przy prawdziwie najtańszych kosztach przelotu między nimi — to znacząca, uczciwa gwarancja, mimo że nie jest jednocześnie dowodem obejmującym przygodne, dodatkowe zbiory wartości.

**Źródło:** A. H. Land, A. G. Doig, "An Automatic Method of Solving Discrete Programming Problems", *Econometrica*, 28(3), 1960 (oryginalna praca o metodzie podziału i ograniczeń); P. Vansteenwegen, W. Souffriau, D. Van Oudheusden, "The Orienteering Problem: A Survey", *European Journal of Operational Research*, 209(1), 2011; [dokumentacja Google OR-Tools](https://developers.google.com/optimization) dla samego solvera CP-SAT.

## 5. Szybkie porównanie

| Klucz | Rodzina | Losowy? | Wybiera na podstawie | Kończy, gdy |
|---|---|---|---|---|
| `greedy` | Zachłanny | Nie | Najbardziej wartościowy sąsiad (remis rozstrzyga koszt) | Brak jakiegokolwiek dostępnego ruchu |
| `direct_to_highest_value` | Zachłanny | Nie | Pojedynczy najbardziej wartościowy cel na siatce | Aktualnie najlepszy cel nieosiągalny |
| `value_cost_ratio` | Zachłanny | Nie | Najlepszy stosunek wartość/koszt w małej, rotującej puli | Brak osiągalnego kandydata w danej rundzie |
| `lowest_cost` | Zachłanny | Nie | Najtańszy w szerokiej, rotującej puli | Brak osiągalnego kandydata w danej rundzie |
| `a_star_to_highest_value` | A* | Nie | Jak `direct_to_highest_value`, trasowanie prawdziwie optymalne | Jak `direct_to_highest_value` |
| `a_star_value_cost_ratio` | A* | Nie | Jak `value_cost_ratio`, trasowanie prawdziwie optymalne | Jak `value_cost_ratio` |
| `a_star_lowest_cost` | A* | Nie | Jak `lowest_cost`, trasowanie prawdziwie optymalne | Jak `lowest_cost` |
| `namoa_star` | A* | Nie | Najlepszy punkt z pełnego menu kompromisów koszt/wartość kandydata | Brak dostępnego kompromisu u żadnego kandydata |
| `ant_colony` | Metaheurystyka | Tak | Feromon + malejące z odległością pole atrakcyjności, na krok | Ustalona liczba iteracji |
| `tabu_search` | Metaheurystyka | Częściowo | Najlepszy, nieobjęty tabu ruch ADD/DROP/SWAP | Ustalone rundy albo brak dopuszczalnego ruchu |
| `variable_neighborhood_search` | Metaheurystyka | Tak | Losowe wstrząśnięcie, potem najlepszy ruch lokalny | Ustalona liczba rund |
| `grasp` | Metaheurystyka | Tak | Losowy wybór spośród najlepszych kandydatów, potem najlepszy ruch lokalny | Ustalona liczba niezależnych powtórzeń |
| `simulated_annealing` | Metaheurystyka | Tak | Jeden losowy ruch, akceptowany/odrzucany wg chłodzącej się temperatury | Ustalona liczba iteracji |
| `genetic_algorithm` | Metaheurystyka | Tak | Populacja + selekcja turniejowa + krzyżowanie + mutacja | Ustalona liczba pokoleń |
| `large_neighborhood_search` | Metaheurystyka | Tak | Adaptacyjna para operatorów wyburzający + naprawiający, akceptacja wg chłodzącej się temperatury | Ustalona liczba rund |
| `exact_solver` | Dokładny | Nie | Każdy podzbiór/kolejność naraz, przez podział i ograniczenia CP-SAT | Dowiedziona optymalność albo limit czasu |

## 6. Pełna bibliografia

- T. H. Cormen, C. E. Leiserson, R. L. Rivest, C. Stein, *Introduction to Algorithms*, MIT Press.
- T. Tsiligirides, "Heuristic Methods Applied to Orienteering", *Journal of the Operational Research Society*, 35(9), 1984.
- B. L. Golden, L. Levy, R. Vohra, "The Orienteering Problem", *Naval Research Logistics*, 34(3), 1987.
- P. E. Hart, N. J. Nilsson, B. Raphael, "A Formal Basis for the Heuristic Determination of Minimum Cost Paths", *IEEE Transactions on Systems Science and Cybernetics*, 4(2), 1968.
- L. Mandow, J. L. Pérez-de-la-Cruz, "A New Approach to Multiobjective A* Search", *Proceedings of IJCAI 2005*.
- M. Dorigo, *Optimization, Learning and Natural Algorithms*, praca doktorska, Politecnico di Milano, 1992.
- M. Dorigo, L. M. Gambardella, "Ant Colony System: A Cooperative Learning Approach to the Traveling Salesman Problem", *IEEE Transactions on Evolutionary Computation*, 1(1), 1997.
- F. Glover, "Future Paths for Integer Programming and Links to Artificial Intelligence", *Computers & Operations Research*, 13(5), 1986.
- F. Glover, M. Laguna, *Tabu Search*, Kluwer Academic Publishers, 1997.
- N. Mladenović, P. Hansen, "Variable Neighborhood Search", *Computers & Operations Research*, 24(11), 1997.
- T. A. Feo, M. G. C. Resende, "Greedy Randomized Adaptive Search Procedures", *Journal of Global Optimization*, 6(2), 1995.
- S. Kirkpatrick, C. D. Gelatt, M. P. Vecchi, "Optimization by Simulated Annealing", *Science*, 220(4598), 1983.
- J. H. Holland, *Adaptation in Natural and Artificial Systems*, University of Michigan Press, 1975.
- D. E. Goldberg, *Genetic Algorithms in Search, Optimization, and Machine Learning*, Addison-Wesley, 1989.
- P. Shaw, "Using Constraint Programming and Local Search Methods to Solve Vehicle Routing Problems", *Proceedings of CP 1998*.
- S. Ropke, D. Pisinger, "An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows", *Transportation Science*, 40(4), 2006.
- A. H. Land, A. G. Doig, "An Automatic Method of Solving Discrete Programming Problems", *Econometrica*, 28(3), 1960.
- P. Vansteenwegen, W. Souffriau, D. Van Oudheusden, "The Orienteering Problem: A Survey", *European Journal of Operational Research*, 209(1), 2011.
- [Dokumentacja Google OR-Tools](https://developers.google.com/optimization) (solver CP-SAT).
