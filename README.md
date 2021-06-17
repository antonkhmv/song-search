# song-search
 
Поисковая система для нахождения песен по загаловку, автору и содержанию

Фильтрует песни по соответствию содержания запроса и ранжирует подходящие результаты по релевантности. 
Для решения я использовал схожесть эмбеддингов текста песен с запросом через consine_simularity -- word2vec для заголовков и автора песни и tdf-idf  для текста песни (для фильтрации - составил обратный индекс).

[!img1](https://github.com/antonkhmv/song-search/imgs/1.jpg)

[!img2](https://github.com/antonkhmv/song-search/imgs/2.jpg)

[!img3](https://github.com/antonkhmv/song-search/imgs/3.jpg)

[!img4](https://github.com/antonkhmv/song-search/imgs/4.jpg)