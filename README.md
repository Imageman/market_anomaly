The script analyzes historical data and searches for anomalies in it. Result of stock market anomalies detection (high result means anomaly).

Example (old) of result:
![sample.png](bin%2Fsample.png)
See interactive graph: https://imageman.github.io/Public/finance_plot.html (more recent)

## Some comments on the project.
- Selected a few important indicators to my liking (such as gold prices, bitcoin rate, summary of the 500 largest US companies, etc.).
- These indicators are available from 2014. Some of them have gaps (like the weekend), which I fill by interpolation.
- I do not train “normality” on all data - the last 45 days are not included in the training, in order not to miss a “big” anomaly. It should also be taken into account that the most recent data are often incomplete and will be refined later. The training of anomaly algorithms involves already refined data. That is why when we look at the anomaly graph, we see some uplift in the latest data. This rise should be treated with some skepticism.

By clicking on the labels in the legend, you can turn on and off individual graphs. The mouse can be used to scale the points of interest. A double mouse click restores the zoom to 100%.

## Некоторые замечания по проекту.
- Выбрано несколько важных показателей по моему вкусу (такие как цены на золото, курс биткоина, сводка по 500 крупнейшим компаниям США и т.п.).
- Эти показатели имеются с 2014 года. В некоторых из них есть пробелы (например выходные), которые я заполняю интерполяцией.
- Обучение "нормальности" я провожу не на всех данных - последние 45 дней не участвуют в обучении, для того, что бы не пропустить "большую" аномалию. Так же следует учесть, что самые последние данные зачастую бывают неполные и они будут уточняться позже. В обучении алгоритмов аномальности участвуют уже уточненные данных. Именно поэтому когда мы смотрим график аномалий мы видим некоторый подъем в последних данных. К этому подъему нужно относится с некоторой долей скептицизма.

При нажатии на надписи в легенде можно отключать и включать отдельные графики. Мышкой можно масштабировать интересующие моменты. Двойной клик мышкой восстанавливает увеличение до 100%