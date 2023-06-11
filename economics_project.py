from bs4 import BeautifulSoup
from IPython.display import HTML
import scrapy
from scrapy.crawler import CrawlerProcess
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.cm as cm
import matplotlib.colors as colors
from geopy.geocoders import Nominatim
import folium
import re
from sympy import symbols, Eq, solve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import networkx as nx
from geopandas.tools import geocode
from geopy.exc import GeocoderTimedOut
import csv

"""# What is the best city?

### В нашем проекте мы будем исследовать города и страны. Мы посмотрим на разные характеристики такие как уровень цен, зарплат, стоимость недвижимости и продуктов в магазине и на основе этих факторов сделаем вывод о том, какой город (или города) – самый лучший. Мы также научимся по стоимости определенного набора продуктов предсказывать цены на кофе в разных городах (поскольку автор этого проекта kindа of obsessed with coffe). 

### Отметим практическую полезность этого проекта: в связи со сложившейся геополитической ситуацией, некоторые люди думают об эмиграции и им важно знать, какие города им подходят и не подходят и почему. 

### Stay tuned! and follow my research...

1) Работа с REST API (XML/JSON) – мы будем загружать данные о зарплатах в разных городах с сайта"https://api.teleport.org/api/urban_areas/", используя API

2) Веб-скреппинг – мы будем загражать данные по странам через Scrapy (см приложенный питоновский файл и полученный выходной файл с данными). Для каких-то локальных маленьких задач, где использование Scrapy излишне, будем пользоваться библиотекой Beautiful Soup

3) Pandas будем пользоваться активно для группировки и трансформации данных

4) Регулярные выражения будем использовать для поиска по сайту определнных паттернов (в нашей задаче будем искать на сайте лучшие города – паттерн "1. {какой-то город}")

5) Математическими возможностями Python будем пользоваться для того, чтобы рассчитать ежемесячный плятеж по ипотеке и сравнить города таким образом. Также посчитаем Housing Affordability для каждого города.

6) Геоданными будем пользоваться для того, чтобы наглядно изобразить на карте города и раскрасить их в зависимости от среднего уровня зарплат. Также будем пользоваться ими для получения координат стран и определния их континента соответственно 

7) Нарисуем граф, в котором будут ребрами соединены страны, расположенные на одном континенте. Подпишем компоненты свзности и в каждой найдем страну, у которой Cost of Living наибольший. 

8) Визуализировать будем.

9) SQL воспользуемся для упрощенных запросов к таблице и узнаем, в каких городах самый дорогой кофе

10) На stremlit проект загрузим

11) Машинное обучение будет использоваться для предсказание цен на кофе, если известны цены на молоко, воду, пиво, поход в ресторан и жилье в городе

12)

Если бы вас спросили, какой город самый лучший, что бы вы сделали в первую очередь? Автор этого проекта решил сильно не напрягаться, ввел в гугл "What is the best city in the world?" и перешел по самой первой ссылке:
"""

# Получим содержимого страницы
response = requests.get('https://www.worldsbestcities.com/rankings/worlds-best-cities/')
soup = BeautifulSoup(response.content, 'html.parser')

# Извлечем текст
text = soup.get_text()

# Найдем города с помощью регулярного выражения
pattern = r'Download Full Report\s+\d\.\s+([A-Za-z\s]*?)(?=\s[^A-Za-z])'
best_cities = re.findall(pattern, soup_text)

# Напечатаем первые 10 городов
print(best_cities[:10])

"""Теперь проведем собственное исследование. Начнем издалека – сравним старны.

C сайта (https://www.numbeo.com/cost-of-living/rankings_by_country.jsp) скачаем данные о странах. Мы сделаем это с помощью библиотеки scrapy, которую запустим локально на компьютере. Код можно посмотреть в дополнительном приложенном файле.

Обзор индексов стоимости жизни на этом сайте: Индексы стоимости жизни, представленные на этом сайте, относятся к Нью-Йорку (NYC), при этом базовый индекс для NYC составляет 100%. Ниже приводится описание каждого индекса и его значение:

1) Cost of Living Index: Этот индекс показывает относительные цены на потребительские товары, такие как продукты питания, рестораны, транспорт и коммунальные услуги. Он не учитывает расходы на проживание, такие как арендная плата или ипотека. Например, город с индексом стоимости жизни 120 оценивается как на 20% более дорогой, чем Нью-Йорк (без учета арендной платы). 

2) Rent Index: Этот индекс оценивает цены на аренду квартир в городе по сравнению с Нью-Йорком. Если индекс арендной платы равен 80, это означает, что средние цены на аренду жилья в этом городе примерно на 20% ниже, чем в Нью-Йорке.

3) Groceries Index: Этот индекс дает оценку цен на продукты питания в городе по сравнению с Нью-Йорком. Numbeo использует весовые коэффициенты товаров из раздела "Рынки" для расчета этого индекса для каждого города.

4) Restaurant Price Index: Этот индекс сравнивает цены на блюда и напитки в ресторанах и барах с ценами в Нью-Йорке.

5) Cost of Living Plus Rent Index: Этот индекс оценивает цены на потребительские товары, включая арендную плату, в сравнении с ценами в Нью-Йорке.

6) Local Purchasing Power Index: Этот индекс показывает относительную покупательную способность в данном городе на основе средней чистой заработной платы. Местная покупательная способность на уровне 40 означает, что жители города со средней зарплатой могут позволить себе в среднем на 60% меньше товаров и услуг по сравнению с жителями Нью-Йорка со средней зарплатой.
"""

df = pd.read_csv('numbeo_data.csv')
df['Country'] = df['Country'].replace('United States', 'United States of America')
print(df['Cost of Living Plus Rent Index'].min())
print(df['Cost of Living Plus Rent Index'].max())

"""Какой огромный разброс в стоимости жизни!"""

df.head()

"""Теперь для каждого континента найдем страну, в которой самый дорогой уровень жизни и построим граф:"""

# Функция для геокодирования с повторными попытками в случае таймаута
def geocode_with_timeout(country):
    try:
        return geocode(country).geometry
    except GeocoderTimedOut:
        return geocode_with_timeout(country)

# Загружаем данные о географическом положении стран
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Геокодируем каждую страну, чтобы получить её координаты
df['geometry'] = df['Country'].apply(geocode_with_timeout)

# Преобразуем датафрейм pandas в GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry='geometry')

# Определяем континент каждой страны
gdf = gpd.sjoin(gdf, world, how="left", op='intersects')
gdf['Continent'] = gdf['continent']
gdf.loc[gdf['Country'] == 'Switzerland', 'Continent'] = 'Europe' # При определении координат происходит ошибка, поэтому нужно вручную добавить информацию

# Удаляем страны, для которых не удалось определить континент
gdf = gdf.dropna(subset=['Continent'])

# Создаем граф
G = nx.Graph()

for _, country in gdf.iterrows():
    G.add_node(country['Country'], continent=country['Continent'], cost_of_living=country['Cost of Living Index'])

for continent in gdf['Continent'].unique():
    countries_in_continent = gdf[gdf['Continent'] == continent]['Country']
    for i, country1 in enumerate(countries_in_continent):
        for country2 in countries_in_continent[i + 1:]:
            G.add_edge(country1, country2)

# Получаем позиции узлов при помощи алгоритма spring layout
pos = nx.spring_layout(G, seed=42)

# Устанавливаем размер фигуры
plt.figure(figsize=(20, 10))

# Рисуем узлы графа
nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')

# Рисуем ребра графа
nx.draw_networkx_edges(G, pos, width=2, edge_color='grey')

# Находим страну с наибольшим индексом стоимости жизни для каждого континента
max_cost_of_living_countries = gdf.loc[gdf.groupby('Continent')['Cost of Living Index'].idxmax()]['Country']

# Рисуем подписи только для стран с наибольшим индексом стоимости жизни
labels = {country: country for country in max_cost_of_living_countries}
nx.draw_networkx_labels(G, pos, labels, font_size=15, font_weight='bold')

# Рисуем подписи для каждого континента
continent_labels = {continent: continent for continent in gdf['Continent'].unique()}
continent_positions = {continent: sum(pos[country] for country in G.nodes if G.nodes[country]['continent'] == continent) / sum(G.nodes[country]['continent'] == continent for country in G.nodes) for continent in continent_labels}

# Смещение для подписей континентов
for continent in continent_positions:
    continent_positions[continent][1] += 0.1

nx.draw_networkx_labels(G, continent_positions, continent_labels, font_size=20, font_color='darkred', font_weight='bold')

plt.show()

"""#### Промежуточный вывод: если вы определились с континентом для переезда, и при этом не хотите вести дорогой образ жизни, то из списка стран стоит вычернуть те, которые представлены выше.

Интересно было бы получить более подробную информацию, поэтому раскрасим на карте страны в зависимости от Cost of Living Plus Rent Index.
"""

df = pd.read_csv('numbeo_data.csv')
df['Country'] = df['Country'].replace('United States', 'United States of America')
# Загрузите геоданные мира
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Слияние данных с геоданными
merged = world.set_index('name').join(df.set_index('Country'))
# Подготовка карты
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.4)

# Рисуем карту
merged.plot(column='Cost of Living Plus Rent Index', 
            cmap='BuGn', 
            linewidth=0.3, 
            ax=ax, 
            edgecolor='0', 
            legend=True, 
            cax=cax,
            vmin = df['Cost of Living Plus Rent Index'].min(),
            vmax = df['Cost of Living Plus Rent Index'].max(),
            missing_kwds={'color': 'steelblue'})
plt.show()

# Выведем названия стран, которые есть в 'world', но отсутствуют в 'df' (они закрашены голубым)
missing_countries_in_df = set(world['name']) - set(df['Country'])
print("Countries present in 'world' but missing in 'df':", missing_countries_in_df)

"""Теперь посмотрим на корреляции между Cost of Living, Rent Index, Restaurant Index Grocery Index и остальными. Предположение заключается в том, что Cost of Living сильнее зависит от Rent Index, чем от Grocery Index."""

corr = df.corr()

plt.figure(figsize=(10, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
plt.title('Корреляционная матрица')
plt.show()

"""Посмотрим на связь между Cost of Living Index и Local Purchasing Power Index. """

sns.set_theme(style="dark", palette=None)
sns.color_palette("icefire", as_cmap=True)

g = sns.jointplot(data=df, x="Cost of Living Index", y="Local Purchasing Power Index", color="steelblue", kind="reg")
g.plot_joint(sns.kdeplot, fill=True, color="steelblue")
sns.scatterplot(data=df, x="Cost of Living Index", y="Local Purchasing Power Index", s=5, color=".15")
sns.kdeplot(data=df, x="Cost of Living Index", y="Local Purchasing Power Index", levels=5, color="steelblue", linewidths=1)

"""#### Видим, что в основном, чем ниже покупательная способность валюты, тем ниже стоимость жизни. В каком-то смысле, это означает, что стоит переезжать в страны со слабыми валютами, если хочется дешево жить. Правда это палка о двух концах – ведь в таких странах вы и зарабатывать будете меньше. Отметим также, что в основном пик стоимости жизни приходится на 30, то есть у стоимость жизни в наибольшем числа стран на 70% ниже, чем в Нью-Йорке. Покупательная способность у наибольшего числа стран на 60% ниже доллара.

Проверим нашу гипотезу о том, что стоимость жизни больше зависит от стоимости жилья нежеле чем от стоимости товаров.
"""

# Добавляем новый столбец 'Index Category'
sns.set_theme(style="dark", palette='icefire')
sns.color_palette("icefire", as_cmap=True)
df['Index Rent Category'] = df['Rent Index'].apply(lambda x: 'High' if x > df['Rent Index'].median() else 'Low')
df['Index Groceries Category'] = df['Groceries Index'].apply(lambda x: 'High' if x > df['Groceries Index'].median() else 'Low')

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

sns.kdeplot(ax=ax[0], data=df, x="Cost of Living Index", y="Local Purchasing Power Index", hue="Index Rent Category", fill=True, color="steelblue")
sns.kdeplot(ax=ax[0], data=df, x="Cost of Living Index", y="Local Purchasing Power Index", hue="Index Rent Category", levels=7, color="b", linewidths=1)
sns.scatterplot(ax=ax[0], data=df, x="Cost of Living Index", y="Local Purchasing Power Index", hue="Index Rent Category", s=2, color=".15")
sns.kdeplot(ax=ax[1], data=df, x="Cost of Living Index", y="Local Purchasing Power Index", hue="Index Groceries Category", fill=True)
sns.kdeplot(ax=ax[1], data=df, x="Cost of Living Index", y="Local Purchasing Power Index", hue="Index Groceries Category", levels=7, color="b", linewidths=1)
sns.scatterplot(ax=ax[1], data=df, x="Cost of Living Index", y="Local Purchasing Power Index", hue="Index Groceries Category", s=2, color=".15")

ax[0].set_title("Rent Index Category")
ax[1].set_title("Groceries Index Category")

plt.tight_layout()
plt.show()

"""Видим, что разницы почти нет. Это достаточно логичный вывод, потому что важна относительная стоимость товаров, а не реальная: отношение цены квартиры к цене кофе различатся от сраны к стране, но находится в определнном промежутке (это также происходит потому что арбитраж возможен только какое-то непродолжительное время, и деньги с неба не падают). Видим, что страны с высокими индексами аренды и товаров гораздо более плавно распределены.

### Один из самых важных вопросов при переезде – вопрос зарплаты. Изучим его.

Для этого при помощи API скачаем данные о зарплатах с сайта https://api.teleport.org/api/urban_areas/.
"""

# Запрос списка городских зон
url = "https://api.teleport.org/api/urban_areas/"
response = requests.get(url)
data = response.json()

# Извлечение ссылок на городские зоны
urban_areas = data['_links']['ua:item']
urban_area_links = [item['href'] for item in urban_areas]

# Создаем пустой датафрейм для сбора информации о зарплатах
df_salaries = pd.DataFrame()

for link in urban_area_links:
    # Запрос данных о зарплатах
    salary_url = link + 'salaries/'
    response = requests.get(salary_url)
    salary_data = response.json()
    
    if 'salaries' in salary_data:
        # Преобразуем данные о зарплатах в датафрейм
        salary_df = pd.json_normalize(salary_data['salaries'])
        
        # Добавим название городской зоны для каждого набора данных о зарплатах
        salary_df['urban_area'] = link.split('/')[-2]  # Извлечение имени городской зоны из ссылки

        df_salaries = df_salaries.append(salary_df, ignore_index=True)
        
df_salaries['urban_area'] = df_salaries['urban_area'].astype(str)
df_salaries['urban_area'] = df_salaries['urban_area'].apply(lambda x: x.replace('slug:', '').strip())

df_salaries.tail()

"""Если переезжает семья, то для того, чтобы оба могли работать, нужно, чтобы эта работа для обоих была, причем достойная! Посмотрим на рапределение зарплат внутри профессий."""

df_salaries['average_salary'] = df_salaries[['salary_percentiles.percentile_25', 
                           'salary_percentiles.percentile_50', 
                           'salary_percentiles.percentile_75']].mean(axis=1)

average_salaries = df_salaries.groupby('urban_area')['average_salary'].mean()

df_salaries['range'] = df_salaries['salary_percentiles.percentile_75'] - df_salaries['salary_percentiles.percentile_25']

# создаем новые столбцы с наибольшим и наименьшим размахом зарплат для каждой профессии
df_salaries['max_range'] = df_salaries.groupby('job.title')['range'].transform('max')
df_salaries['min_range'] = df_salaries.groupby('job.title')['range'].transform('min')

# сортируем по максимальному и минимальному размаху и выводим первые пять профессий
top_5_titles = df_salaries.sort_values('max_range', ascending=False)['job.title'].drop_duplicates()[:5]
bottom_5_titles = df_salaries.sort_values('min_range')['job.title'].drop_duplicates()[:5]

top_5_urban_area = df_salaries.loc[df_salaries['max_range'].nlargest(5).index, 'urban_area'].unique()
bottom_5_urban_area = df_salaries.loc[df_salaries['min_range'].nsmallest(5).index, 'urban_area'].unique()

# создаем списки из полученных профессий
top_5_titles = top_5_titles.tolist()
bottom_5_titles = bottom_5_titles.tolist()

print("Высокий разброс зарплат:")
for title in top_5_titles:
    df_title = df_salaries[df_salaries['job.title'] == title]
    max_salary_location = df_title['urban_area'][df_title['salary_percentiles.percentile_50'].idxmax()]
    min_salary_location = df_title['urban_area'][df_title['salary_percentiles.percentile_50'].idxmin()]
    print(f"Для профессии {title}, максимальная зарплата в {max_salary_location}, минимальная зарплата в {min_salary_location}")

print("\nНизкий разброс зарплат:")
for title in bottom_5_titles:
    df_title = df_salaries[df_salaries['job.title'] == title]
    max_salary_location = df_title['urban_area'][df_title['salary_percentiles.percentile_50'].idxmax()]
    min_salary_location = df_title['urban_area'][df_title['salary_percentiles.percentile_50'].idxmin()]
    print(f"Для профессии {title}, максимальная зарплата в {max_salary_location}, минимальная зарплата в {min_salary_location}")

"""#### Кажется, переезжать в Havana не самая лучшая идея :)

С помощью карт получим более широкую картину. (Кстати, на этой карте можно наводить и смотреть среднюю зарплату для каждого конкретного города – попробуйте. Зарплаты указаны в долларах.)
"""

# Инициализация геолокатора
geolocator = Nominatim(user_agent="myGeocoder")

# Нормализация зарплат
norm = colors.Normalize(vmin=average_salaries.min(), vmax=average_salaries.max())
# Создаем цветовую карту
colormap = cm.get_cmap("YlOrRd")

# Создаем карту
m = folium.Map(location=[46.8182, 8.2275], zoom_start=7)

# Добавляем маркеры для каждого города
for idx, row in average_salaries.reset_index().iterrows():
    # Получаем координаты города
    location = geolocator.geocode(row['urban_area'])
    
    # Если координаты не найдены, пропускаем итерацию
    if not location:
        continue

    # Создаем цвет на основе нормализованной средней зарплаты
    rgb_color = colormap(norm(row['average_salary']))[:3]
    hex_color = colors.rgb2hex(rgb_color)
    
    # Добавляем круглый маркер на карту
    folium.CircleMarker(
        location=[location.latitude, location.longitude],
        radius=10,
        popup=row['urban_area'] + ': $' + str(round(row['average_salary'], 2)),
        color=hex_color,
        fill=True,
        fill_color=hex_color
    ).add_to(m)

# Выводим карту
m

"""#### Если хочется много зарабатывать, поехать в Америку – отличная идея. В Европе по уровню зарплат лидирует Цюрих.

### Сторона расходов. Жилье

Теперь обратимся к стороне трат и изучим подробнее стоимости товаров и жилья. C того же сайта numbeo скачаем данные о товарах для тех городов, для которых они есть, удаляя города, которые нужно было бы обрабатывать вручную. (Мы просто скопируем список городов вручную и сохраним в csv файл.)
"""

cities = []
with open('cities.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Пропускаем заголовок, если он есть
    for row in reader:
        row[0] = row[0].replace(' ', '-')
        if (row[0] == 'Kiev-(Kyiv)'):
            row[0] = 'Kiev'
        if (row[0] == 'Krakow-(Cracow)'):
            row[0] = 'Krakow'
        if (row[0] == 'Astana-(Nur-Sultan)'):
            row[0] = 'Astana-Nur-Sultan-Kazakhstan'
        if (row[0] == 'The-Hague-(Den-Haag)'):
            row[0] = 'The-Hague-Den-Haag-Netherlands'
        cities.append(row[0])

# Удаляем эти города, потому что они плохо парсятся
cities.remove('Gurgaon')
cities.remove('Noida')
cities.remove('Delhi')
cities.remove('Pune')
cities.remove('Mumbai')
cities.remove('Bangalore')
cities.remove('Hyderabad')
cities.remove('Jaipur')
cities.remove('Ahmedabad')
cities.remove('Kolkata')
cities.remove('Chennai')
print(cities)

def scrape_numbeo(url, city):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    table = soup.find('table', {'class': 'data_wide_table new_bar_table'})
    if table is None:
        print(f"Couldn't find the table on {url}. Skipping this city...")
        return None, None
    headers = []
    for th in table.findAll('th'):
        headers.append(th.text.strip())

    data = []
    for row in table.findAll('tr'):
        columns = row.findAll('td')
        output_row = {}
        for header, column in zip(headers, columns):
            output_row[header] = column.text.strip()
        if output_row:
            data.append(output_row)

    df = pd.DataFrame(data)
    
    df_edit = df[['Restaurants', 'Edit']].set_index('Restaurants').transpose()
    df_edit.index = [city]
    df_edit.columns.name = 'Cities'
    
    df_range = df[['Restaurants', 'Range']].set_index('Restaurants').transpose()
    df_range.index = [city]
    df_range.columns.name = 'Range of Price'
    
    return df_edit, df_range

base_url = 'https://www.numbeo.com/cost-of-living/in/'
df_edit = pd.DataFrame()
df_range = pd.DataFrame()

for city in cities:
    city_url = base_url + city + '?displayCurrency=USD'
    city_df_edit, city_df_range = scrape_numbeo(city_url, city)
    df_edit = pd.concat([df_edit, city_df_edit])
    df_range = pd.concat([df_range, city_df_range])

for col in df_edit.columns:
    df_edit[col] = df_edit[col].replace({'\$': '', ',': '', '\xa0': '', '€': ''}, regex=True).astype(float)

df_edit.head()

# Создаем символьные переменные
affordability_ratio = symbols('affordability_ratio')

# Рассчитываем среднее значение 'Housing Affordability' без учета 'affordability_ratio'
average_affordability = np.mean(df_edit['Average Monthly Net Salary (After Tax)'] / df_edit['Apartment (1 bedroom) in City Centre'])

# Решаем уравнение относительно 'affordability_ratio'
equation = Eq(average_affordability / affordability_ratio, 1)
solution = solve(equation, affordability_ratio)

# Если решение существует, добавляем столбец 'Housing Affordability' в DataFrame
if solution:
    df_edit['Housing Affordability'] = df_edit['Average Monthly Net Salary (After Tax)'] / (df_edit['Apartment (1 bedroom) in City Centre'] * float(solution[0]))

    # Сортируем DataFrame по столбцу 'Housing Affordability' по убыванию
    df_sorted = df_edit.sort_values(by='Housing Affordability', ascending=False)

    # Выводим топ-10 городов с наивысшим показателем 'Housing Affordability'
    print("Top 10 cities with highest 'Housing Affordability':")
    for index, row in df_sorted.head(10).iterrows():
        print(f"{index} - {row['Housing Affordability']}")
        
    print()
    
    # Выводим топ-10 городов с наименьшим показателем 'Housing Affordability'
    print("Top 10 cities with lowest 'Housing Affordability':")
    for index, row in df_sorted.tail(10).iterrows():
        print(f"{index} - {row['Housing Affordability']}")

"""#### Крайне интересные результаты! Автор этого проекта сам живет в Вене и может подтвердить, что буквально за 700-1000 евро тут можно снять квартиру очень хорошего качества (с практически дизайнерским интерьером), причем в центре. К слову, плохая квартира в Москве (не в центре) примерно столько же стоит. В целом полезное наблюдение, что в Осло, Вене, Чикаго, Брюсселе и др. квартиры крайне доступные и уровень жизни в этих городах тоже достойный.

#### Автора удивило, что наименее доступное жилье в "дешевых" городах: Стамбуле, Ташкенте, Ереване, Белграде и др. Из нашей формулы расчета следует, что в этих городах либо очень низкие зарплаты, либо очень дорогое жилье само по себе. Вероятно, зарплаты настолько низкие, что относительная (относительно Нью-Йорка или Лондона) недороговизна самого жилья не является приимуществом этих городов. (Впрочем, в современном мире часто можно работать онлайн и жить где угодно, поэтому конкретну эту часть нашего исследование стоило бы в дальнейшем уточнять.)

Посмотрим значение Housing Affordability для городов, которые мы в самом начале проекта определеили как лучшие города:
"""

best_cities = [city.replace("New York", "New-York") for city in best_cities]

affordability_values = df_edit.loc[best_cities, 'Housing Affordability']

# Выводим данные в формате "город - значение Housing Affordability"
for city, value in affordability_values.items():
    print(f"{city} - {value}")

"""Теперь предположим, что мы хотим квартиру не просто снимать, а хотим ее купить. Очень вероятно, нам потребуется взять иппотеку. Посмотрим на размер платежа в разных странах и сравним.

Для расчета этого значения мы будем использовать следующую формулу:

P = [r * PV] / [1 - (1 + r)^ - n]

где:

P - ежемесячный платеж
r - месячная ставка ипотеки (годовая ставка / 12 / 100)
PV - сумма ипотечного кредита (принимаем за стоимость квартиры "Price per Square Meter to Buy Apartment in City Centre", считаем, что мы покупаем квартиру 50 кв.м.)
n - количество платежей (предположим, что это 20 лет, то есть 20*12 месяцев)
"""

# Символы для переменных
P, r, PV, n = symbols('P r PV n')

# Срок кредита в месяцах
n_value = 20 * 12

# Добавление нового столбца "Mortgage Payment"
df_edit['Mortgage Payment'] = 0

# Расчет ежемесячного платежа по ипотеке для каждого города
for i, row in df_edit.iterrows():
    # Ипотечная формула
    mortgage_formula = Eq(P, (r*PV) / (1 - (1 + r)**-n))
    r_value = row['Mortgage Interest Rate in Percentages (%), Yearly, for 20 Years Fixed-Rate'] / 12 / 100
    PV_value = row['Price per Square Meter to Buy Apartment in City Centre'] * 50 # Предположим, что цена указана в миллионах
    monthly_payment = solve(mortgage_formula.subs({PV: PV_value, r: r_value, n: n_value}), P)
    df_edit.loc[i, 'Mortgage Payment'] = float(monthly_payment[0])

# Сортировка по столбцу 'Mortgage Payment'
df_sorted = df_edit.sort_values(by='Mortgage Payment')

# Вывод топ-10 городов с наименьшим ежемесячным платежом
print("Топ-5 городов с наименьшим ежемесячным платежом:")
for index, row in df_sorted.head(10).iterrows():
    print(f"Город: {index}, Ежемесячный платеж: {row['Mortgage Payment']}")

# Вывод топ-10 городов с наибольшим ежемесячным платежом
print("\nТоп-5 городов с наибольшим ежемесячным платежом:")
for index, row in df_sorted.tail(10).iterrows():
    print(f"Город: {index}, Ежемесячный платеж: {row['Mortgage Payment']}")

# Вывод ежемесячных платежей для городов из best_cities
print("\nЕжемесячные платежи для выбранных городов:")
for city in best_cities:
    payment = df_edit[df_edit.index == city]['Mortgage Payment'].values[0]
    print(f"Город: {city}, Ежемесячный платеж: {payment}")

"""#### Видим, что в Лондон, Нью-Йорк и Цюрих вошли в топ городов с самой высокой стоимостью ипотечного платежа (не очень удивительно).

### Сторона расходов. Товары

А теперь давайте создадим SQL-табличку и будем задавать ей глупые (и не только) вопросы, все-все, которые приходят на ум!
"""

df = df_edit.reset_index().rename(columns={'index': 'Cities'})
df.columns = df.columns.str.replace('(', '')
df.columns = df.columns.str.replace(')', '')
df.columns = df.columns.str.replace(',', '')
df.columns = df.columns.str.replace('-', '_')
df.columns = df.columns.str.replace(' ', '_')

# Нам нужно убрать дублирующиеся колонки, чтобы SQL-табличка собралась
def deduplicate_column_names(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique(): 
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df

df = deduplicate_column_names(df)
df.head()

engine = create_engine('sqlite:///city_db.sqlite')

# Сохранение DataFrame в SQL таблицу "City_Data" с указанным именем для индекса
df.to_sql('City_Data', engine, if_exists='replace')

# Commented out IPython magic to ensure Python compatibility.
# %load_ext sql
# %sql sqlite:///city_db.sqlite

"""Итак, в каких 5 городах самый дорогой кофе?"""

# Commented out IPython magic to ensure Python compatibility.
# %%sql
# SELECT Cities, Cappuccino_regular
# FROM City_Data
# ORDER BY Cappuccino_regular DESC
# LIMIT 5

"""А в каких трех городах самый дешевый кофе?"""

# Commented out IPython magic to ensure Python compatibility.
# %%sql
# SELECT Cities, Cappuccino_regular
# FROM City_Data
# ORDER BY Cappuccino_regular ASC
# LIMIT 3

"""В каких городах выгоднее жилье купить, нежели чем снимать?
Для этого нужно определить метрику, которая определяет, что значит "выгоднее купить, нежели снимать": будем считать, что это означает, что стоимость покупки квартиры меньше, чем сумма аренды за определенный период времени (например, 10 лет), здесь я предполагаю, что размер квартиры - 50 кв. метров,:
"""

# Commented out IPython magic to ensure Python compatibility.
# %%sql
# SELECT Cities
# FROM City_Data
# WHERE Price_per_Square_Meter_to_Buy_Apartment_in_City_Centre*50 < Apartment_1_bedroom_in_City_Centre*12*10

"""Смотрите, мы получили, что только в 6(!) городах реально выгоднее жилье купить. Впрочем, это известный экономический результат. Если добавить к этому частые переезды по работе (и не только по работе) покупка квартиры в принципе выглядит не самой удачной идеей.

В каких 5 городах дороже всего сходить в ресторан и сколько это стоит?
"""

# Commented out IPython magic to ensure Python compatibility.
# %%sql
# SELECT Cities, Meal_for_2_People_Mid_range_Restaurant_Three_course
# FROM City_Data
# ORDER BY Meal_for_2_People_Mid_range_Restaurant_Three_course DESC
# LIMIT 5;

"""А дешевле всего?"""

# Commented out IPython magic to ensure Python compatibility.
# %%sql
# SELECT Cities, Meal_for_2_People_Mid_range_Restaurant_Three_course
# FROM City_Data
# ORDER BY Meal_for_2_People_Mid_range_Restaurant_Three_course ASC
# LIMIT 5;

"""В каких 5 городах дороже всего сходить в Макдональдс?"""

# Commented out IPython magic to ensure Python compatibility.
# %%sql
# SELECT Cities, McMeal_at_McDonalds_or_Equivalent_Combo_Meal
# FROM City_Data
# ORDER BY McMeal_at_McDonalds_or_Equivalent_Combo_Meal DESC
# LIMIT 5;

"""А дешевле всего?"""

# Commented out IPython magic to ensure Python compatibility.
# %%sql
# SELECT Cities, McMeal_at_McDonalds_or_Equivalent_Combo_Meal
# FROM City_Data
# ORDER BY McMeal_at_McDonalds_or_Equivalent_Combo_Meal ASC
# LIMIT 5;

"""#### Мы получили, что стоимость самых дешевых походов в хороший ресторан практически равна максимальной стоимости похода в Макдональдс. Также видим, что самая дешевая стоимость похода в Макдональдс меньше самой дорогой стоимости кофе. 

#### Попробуем научиться предсказывать стоимость кофе.
"""

df_edit.head()

# Предобработка данных
data = df_edit
data = data.dropna()  # удаляем пропущенные значения

# Определяем X (входные данные) и y (то, что мы хотим предсказать)
X = data.drop('Cappuccino (regular)', axis=1)
y = data['Cappuccino (regular)']

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем словарь с моделями
models = {
    'linear_regression': LinearRegression(),
    'ridge': Ridge(),
    'lasso': Lasso(),
    'random_forest': RandomForestRegressor(n_estimators=100),
    'gradient_boosting': GradientBoostingRegressor(),
    'svr': SVR(),
    'decision_tree': DecisionTreeRegressor(),
    'knn': KNeighborsRegressor(),
}

# Обучаем модели и оцениваем их точность
for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{name} score: {score*100:.2f}%")

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Создаем словарь с моделями
models = {
    'linear_regression': LinearRegression(),
    'ridge': Ridge(),
    'lasso': Lasso(),
    'random_forest': RandomForestRegressor(n_estimators=100),
    'gradient_boosting': GradientBoostingRegressor(),
    'svr': SVR(),
    'decision_tree': DecisionTreeRegressor(),
    'knn': KNeighborsRegressor(),
}

# Обучаем модели и оцениваем их точность
for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{name} score: {score*100:.2f}%")

"""Как только мы поменяли размер теста с 0.2 на 0.1, начали лучше работать линейная, лассо и ридж регрессии. Видим, что лучшие предсказания дают random_forest и gradient_boosting.

#### Интересно было бы в целом посмотреть на разброс стоимостей товаров в разных городах.
"""

# Выбираем подмножество переменных
selected_categories_food = ['Meal, Inexpensive Restaurant', 
                       'McMeal at McDonalds (or Equivalent Combo Meal)', 
                       'Loaf of Fresh White Bread (500g)']
selected_categories_drink = [ 'Cappuccino (regular)',
                       'Milk (regular), (1 liter)',
                       'Coke/Pepsi (0.33 liter bottle)', 
                       'Water (0.33 liter bottle)']

# Создаем новый DataFrame только с выбранными переменными
df_selected_food = df_edit[selected_categories_food]
df_selected_drink = df_edit[selected_categories_drink]

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="whitegrid", palette="YlOrBr", rc=custom_params)

# Построение графика распределения цен на напитки
plt.figure(figsize=(9,6))
sns.violinplot(data=df_selected_food, palette="YlOrBr", orient="h", scale="count", bw=.2, cut=1, linewidth=1)
plt.title('Распределение цен на еду')
plt.xlabel('Стоимость в $')
sns.despine(left=True, bottom=True)
plt.show()

selected_categories = ['Meal for 2 People, Mid-range Restaurant, Three-course']

# Создаем новый DataFrame только с выбранными переменными
df_selected = df_edit[selected_categories]

sns.violinplot(data=df_selected, inner = 'quartile', palette="YlOrBr", orient="h", scale="count", bw=.2, cut=1, linewidth=1)
plt.title('Распределение цен на рестораны')
sns.despine(left=True, bottom=True)
plt.show()

# Построение графика распределения цен на напитки
plt.figure(figsize=(9,6))
sns.violinplot(data=df_selected_drink, palette="YlOrBr", orient="h", scale="count", bw=.2, cut=1, linewidth=1)
plt.title('Распределение цен на напитки')
plt.xlabel('Стоимость в $')
sns.despine(left=True, bottom=True)
plt.show()

"""Видим, что стоимости напитков колеблются значительно меньше стоимостей еды и уж тем более меньше стоимости похода в хороший ресторан.

## Выводы. 



## Ехать нужно в Чикаго. Там одноврменно высокие зарплаты и доступное жилье.

## P.S. не является инвестиционной рекомендацией, так как при принятии решения о переезде нужно смотреть на такие параметры как состояние экологии, образование и медицина. Все эти параметры в нашем исследовании не рассматривались. 

## P.P.S. Кофе получится купить более-менее везде, потому что цены на него колеблются в разумных пределах.
"""
