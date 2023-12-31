# Bystrenina_Kompozit

С помощью фреймворка Flask было разработано одностраничное пользовательское веб-приложение, прогнозирующее ряд конечных свойств получаемых композиционных материалов на основе нейронной сети. Для запуска приложения пользователь должен перейти по ссылке на сайт: http://127.0.0.1:5000/ . Flask-приложение представляет собой форму, состоящую из одиннадцати входов, куда вводятся значения ряда параметров: соотношение матрица-наполнитель; плотность, кг/м3; модуль упругости, ГПа; количество отвердителя, м.%; содержание эпоксидных групп,%; температура вспышки, С_2; поверхностная плотность, г/м2; потребление смолы, г/м2; угол нашивки, град; шаг нашивки; плотность нашивки. После этого нужно нажать на кнопку «Отправить», и модель выдаст прогнозное значение модуля упругости от растяжения, прочности при растяжении при заданных параметрах.

Содержание работы:

Введение	3

1.Аналитическая часть	5

1.1. Постановка задачи	5

1.2. Описание используемых методов	5

1.3. Разведочный анализ данных	8

1.4. Гистограммы	10

1.5. Попарные графики рассеивания	11

1.6. Визуализация распределения по целевым переменным, выбросы	12

1.7. Матрица корреляции признаков	14

1.8. Диаграмма "Ящик с усами" (Boxplot)	15

2 Предобработка данных	17

2.1 Удаление выбросов	17

2.2 Нормализация	18

2.3 Описательная статистика характеристик после предобработки	19

3  Модели для прогноза целевых переменных	20

3.1  Линейная регрессия	23

3.2 Регрессия k-ближайших соседей	26

3.3 Метод регрессии опорных векторов SVR	28

3.4 Random Forest Regressor	29

4  Рекомендательная нейросеть для соотношения матрица-наполнитель	32

5 Создание Flask приложения	34

Заключение	38

Список использованной литературы	39

