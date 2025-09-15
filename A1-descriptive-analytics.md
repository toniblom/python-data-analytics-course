# Python data analytics - Assignment 1 - descriptive analytics

Author: Toni Blom

Created: 2025-09-15

Original [Google Colab file](https://colab.research.google.com/drive/1Vzr6uneZcaf0LvN0MVnRhBa1fgQoOau7?usp=sharing) (in Finnish).

This assignment has two parts:
* A dataset based on the passengers of Titanic
* A dataset of Finnish municipalities including various demographic and economic indicators

## Preparing the environment

```python
# Connect Colab to Google Drive
from google.colab import drive
drive.mount('/gdrive')
%cd /gdrive/MyDrive/data
```

<img width="268" height="62" alt="image" src="https://github.com/user-attachments/assets/7711c4fc-d213-4a7b-8cac-e1d03dd85876" />

```python
# Import libraries needed in the assignment
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
```

## Part 1: Titanic

```python
# Import titanic data to Colab from Excel file
dft = pd.read_excel('titanic.xlsx')
```

Let's get take a look at what the data set looks like:

```python
# Show the first 5 rows of data
dft.head()
```

<img width="856" height="237" alt="image" src="https://github.com/user-attachments/assets/fb2b013a-2905-4cfc-b86a-77d67b175709" />

```python
# How big is the data set? 1309 rows and 14 columns
dft.shape
```

<img width="105" height="26" alt="image" src="https://github.com/user-attachments/assets/ef0a9b27-9bc8-4bda-9bb6-b2a692068b94" />

```python
# See information about the columns
dft.info()
```

<img width="294" height="311" alt="image" src="https://github.com/user-attachments/assets/953985d9-3641-46d6-9b9d-4e97d853c533" />

Of the columns used in this assignment, the age column has some missing values.

Let's try filtering the data:

```python
# Show male passengers that embarked in Cherbourg, travelled in 2nd class and survived
dft[(dft['embarked']=='C')&(dft['survived']==1)&(dft['sex']=='male')&(dft['pclass']==2)]
```

<img width="858" height="227" alt="image" src="https://github.com/user-attachments/assets/eac27255-b647-47f9-9eba-5e1187484320" />

### Gender distribution of the passengers

Let's examine the gender distribution of the passengers:

```python
# Create a frequency table of gender distribution
dft1 = pd.crosstab(dft['sex'], 'f')

# Translate row headings into Finnish
sukupuoli = ['Nainen', 'Mies']
dft1.index = sukupuoli

# Add a percentage column
n_sex = dft1['f'].sum()
dft1['%'] = dft1['f']/n_sex*100

# Some basic styling
dft1.columns.name = ''
dft1.style.format({'f' : '{:.0f}', '%' : '{:.1f}'})
```

<img width="156" height="86" alt="image" src="https://github.com/user-attachments/assets/e6573073-0903-419a-93bd-b43112676161" />

```python
# Show passengers' gender distribution graphically
dft1['%'].plot(kind='barh')
plt.grid(axis='x')
plt.xlabel(f'%, n = {n_sex}')
plt.show()
```

<img width="498" height="352" alt="image" src="https://github.com/user-attachments/assets/f6be8282-5a79-4b41-ac0d-1271874d82cd" />

* 64.4% of the passengers were male and 35.6% female.

### Age distribution of the passengers

Let's examine the age distribution of the passengers:

```python
# Let's take a look at statistical key values of the passengers' ages
dft['age'].describe()
```

<img width="170" height="266" alt="image" src="https://github.com/user-attachments/assets/dd39269e-37b3-488d-8b7c-8e31ee169e1b" />

* The average age of the passengers was 29.9 years.
* The median age of the passengers was 28 years.
* The youngest passenger was 0.17 years old and the oldest passenger 80 years old.

Let's create a categorical variable 'ikäluokka' (= age category) based on age column:

```python
# Define the limits of the age categories
# Based on the minimum age (0.17) and maximum age (80), choose the following limits
rajat = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

# Add a new column into the data frame
dft['ikäluokka'] = pd.cut(dft['age'], bins=rajat, right=False)
dft.head()
```

<img width="857" height="356" alt="image" src="https://github.com/user-attachments/assets/1e3e636f-7208-46bf-aaab-54366e7a1c72" />

```python
# Create a frequency table of age categories and add a percentage columns
dft2 = pd.crosstab(dft['ikäluokka'], 'f')
dft2.columns.name = ''
n_age = dft2['f'].sum()
dft2['%'] = dft2['f']/n_age*100
dft2.loc['yhteensä'] = dft2.sum()
dft2.style.format({'f' : '{:.0f}', '%' : '{:.1f}'})
```

<img width="193" height="310" alt="image" src="https://github.com/user-attachments/assets/7de9ffde-6545-4fe0-84ed-74df5a0bc9f3" />

```python
# Create a histogram showing the age distribution of the passengers
sns.histplot(dft['age'], bins=rajat)
plt.xlabel('Ikä')
plt.ylabel(f'Lukumäärä, n = {n_age}')
plt.grid(axis='y')
plt.show()
```

<img width="491" height="354" alt="image" src="https://github.com/user-attachments/assets/5141b256-382e-4829-99e2-b883bd8e5a43" />

* The most common age category among the passengers was 20 - 30 years (32.9%).

### Passenger classes and survival rate

Let's examine how the passengers were distributed into different passenger classes:

```python
# Create a frequency table of passenger class
dft3 = pd.crosstab(dft['pclass'], 'f')

# Add a percentage column
n_pclass = dft3['f'].sum()
dft3['%'] = dft3['f']/n_pclass*100

# Styling
dft3.columns.name = ''
dft3.style.format({'f' : '{:.0f}', '%' : '{:.1f}'})
```

<img width="160" height="140" alt="image" src="https://github.com/user-attachments/assets/dfd4d5f6-62be-4957-9d78-ae4afc28e1a9" />

* 323 passengers (24.7%) traveled in 1st class.
* 277 passengers (21.2%) traveled in 2nd class.
* 709 passengers (54.2%) traveled in 3rd class.

Let's examine the rate of survival of all passengers:

```python
# Create a normalized frequency table
pd.crosstab(dft['survived'], 'f', normalize='columns')*100
```

<img width="170" height="117" alt="image" src="https://github.com/user-attachments/assets/f997f7fa-5b60-4e30-aaae-4e5eb9728e4b" />

* 38.2% of all passengers survived, 61.8% did not.

Let's examine how the passenger class affected survival rate:

```python
# Crosstabulate passenger class and survival, normalize by passenger class
dft4 = pd.crosstab(dft['pclass'], dft['survived'], normalize='index')*100

# Translate rows and columns into Finnish
selviytyminen = ['Ei selviytynyt', 'Selviytynyt']
dft4.columns = selviytyminen
n_1 = dft3.loc[1,'f']
n_2 = dft3.loc[2,'f']
n_3 = dft3.loc[3,'f']
dft4.index = [f'Luokka 1, n = {n_1}', f'Luokka 2, n = {n_2}', f'Luokka 3, n = {n_3}']

dft4.style.format('{:.1f} %')
```

<img width="326" height="112" alt="image" src="https://github.com/user-attachments/assets/c3c7994c-75fc-48b7-b6e8-cff9f59d0d67" />

```python
# Show survival rates of different passenger classes graphically
dft4.plot(kind='barh', stacked=True, cmap='copper')
plt.xlim([0, 100])
plt.xticks([0,10,20,30,40,50,60,70,80,90,100])
plt.grid(axis='x')
plt.legend(loc=[0,-0.25], ncols=2)
plt.xlabel('Prosenttia matkustajaluokasta')
plt.show()
```

<img width="567" height="394" alt="image" src="https://github.com/user-attachments/assets/eb4354cf-e5cd-46fa-a056-5ad60141c89f" />

* Passenger class did seem to have an effect on survival rates. Passengers traveling in 1st class had a higher survival rate (61.9%) than either 2nd class (43.0%) or 3rd class (25.5%) passengers.

### Gender and survival rate

Let's examine whether gender affected survival rate:

```python
# Crosstabulate gender and survival, normalize by gender
dft5 = pd.crosstab(dft['sex'], dft['survived'], normalize='index')*100
n_female = dft1.loc['Nainen', 'f']
n_male = dft1.loc['Mies', 'f']
dft5.index = [f'Nainen, n = {n_female}', f'Mies, n = {n_male}']
dft5.columns = selviytyminen
dft5.style.format('{:.1f} %')
```

<img width="306" height="86" alt="image" src="https://github.com/user-attachments/assets/72bbaf89-ac96-49b6-994b-84254ba16888" />

```python
# Show survival rates based on gender graphically
dft5.plot(kind='barh', stacked=True, cmap='copper')
plt.xlim([0, 100])
plt.xticks([0,10,20,30,40,50,60,70,80,90,100])
plt.grid(axis='x')
plt.legend(loc=[0,-0.25], ncols=2)
plt.xlabel('Prosenttia sukupuolesta')
plt.show()
```

<img width="554" height="391" alt="image" src="https://github.com/user-attachments/assets/cab84219-0813-42ab-8076-29dda80c0002" />

* Female passengers had a higher survival rate (72.7%) compared to male passengers (19.1%).

Let's examine gender distributions in different passenger classes:

```python
# Crosstabulate passenger class and gender, normalize by passenger class
dft6 = pd.crosstab(dft['pclass'], dft['sex'], normalize='index')*100
dft6.style.format('{:.1f} %')
```

<img width="191" height="139" alt="image" src="https://github.com/user-attachments/assets/23e6f072-0ec1-4255-a995-c37f21e7cef1" />

* There was a higher percentage of female passengers in the 1st class (44.6%) compared to either 2nd class (38.3%) or 3rd class (30.5%).

## Part 2: Finnish Municipalities

Let's get take a look at what the data set looks like:

```python
# Load the data to Colab and show the first 5 rows
dfk = pd.read_excel('kunnat.xlsx')
dfk.head()
```

<img width="841" height="238" alt="image" src="https://github.com/user-attachments/assets/94202901-9051-4285-84c3-bbedf3958dd9" />

```python
# What is the size of the dataset? 309 rows, 33 columns
dfk.shape
```

```python
# Show information on the columns
dfk.info()
```

<img width="758" height="573" alt="image" src="https://github.com/user-attachments/assets/88e852ad-1cbb-43a0-819e-d44bb42b9d9f" />

* There are no missing values in the data set.
* The data set headers will be translated in this report when necessary.

### Filtering

Let's examine the smallest municipalities by population count ("Väkiluku, 2021") in Finland:

```python
# Show 5 smallest municipalities by population count
dfk.nsmallest(5, 'Väkiluku, 2021')
```

<img width="866" height="237" alt="image" src="https://github.com/user-attachments/assets/c1cdb3a5-e747-4625-bcf4-fa0c5c8e9d56" />

* Many of the smallest municipalities in Finland are located in the Åland archipelago. These municipalities also have a high percentages of Swedish speaking population ("Ruotsinkielisten osuus väestöstä, %, 2021).

Let's filter the smallest municipalities by population count where the percentage of Swedish speaking population is less than 50%.

```python
# Filter 5 smallest municipalities by population count where Swedish speaking population is less than 50%
dfk[dfk['Ruotsinkielisten osuus väestöstä, %, 2021']<50].nsmallest(5, 'Väkiluku, 2021')
```

<img width="864" height="241" alt="image" src="https://github.com/user-attachments/assets/1b9c124b-b250-4711-824a-634b133d3541" />

Let's examine the municipalities where the population has been growing the most ("Väkiluvun muutos edellisestä vuodesta, %, 2021"):

```python
# Filter 5 municipalities with the highest population growth
dfk.nlargest(5, 'Väkiluvun muutos edellisestä vuodesta, %, 2021')
```

<img width="856" height="235" alt="image" src="https://github.com/user-attachments/assets/7912a753-3705-4a77-a3cf-332eb0634e6d" />

Let's examine the municipalities with highest percentages of foreign citizens ("Ulkomaan kansalaisten osuus väestöstä, %, 2021"):

```python
# Filter 5 municipalities with the highest percentage of foreign citizens
dfk.nlargest(5, 'Ulkomaan kansalaisten osuus väestöstä, %, 2021')
```

<img width="857" height="238" alt="image" src="https://github.com/user-attachments/assets/49ae8513-68d1-44fb-bbf8-f01d7bbd1bd7" />

* The highest percentages of foreign citizens are in municipalities with small population counts in the Åland archipelago.

Let's examine the municipalities with highest percentages of foreign citizens where the total population count is over 1000 and the percentage of Swedish speaking population is less than50 %:

```python
# Filter 5 municipalities with the highest percentage of foreign citizens where total population count is over 1000 and Swedish speaking population is less than 50%
dfk[(dfk['Ruotsinkielisten osuus väestöstä, %, 2021']< 50)&(dfk['Väkiluku, 2021']>=1000)].nlargest(5, 'Ulkomaan kansalaisten osuus väestöstä, %, 2021')
```

<img width="856" height="238" alt="image" src="https://github.com/user-attachments/assets/d8fdecf8-32fe-4325-9b3a-65fcb411ad09" />

* Many of the municipalities in this filtering are located in the capital region of Finland.

Let's examine the municipalities with highest social and health service expenditures per capita ("Sosiaali- ja terveystoiminta yhteensä, nettokäyttökustannukset, euroa/asukas, 2020").

```python
# Filter 5 municipalities with highest social and health service expenditures per capita
dfk.nlargest(5, 'Sosiaali- ja terveystoiminta yhteensä, nettokäyttökustannukset, euroa/asukas, 2020')
```

* Many of these municipalities are located in Northern Finland.

### Statistical key values and distributions

#### Population count

Let's examine statistical key values of the population counts of all municipalities:

```python
# Show statistical key values of population counts
dfk['Väkiluku, 2021'].describe()
```

<img width="188" height="266" alt="image" src="https://github.com/user-attachments/assets/532cd30b-a781-4dbb-ad55-7b8d952c5104" />

* The average population count is 17 955, median 5 967, minimum 105 and maximum 658 547.

Let's examine the distribution of population counts with a histogram:

```python
# Create histogram showing population count distributions in different municipalities
sns.histplot(data=dfk['Väkiluku, 2021'])
plt.ylabel(f'Lukumäärä, n = {dfk.shape[0]}')
plt.grid(axis='x')
```

<img width="496" height="349" alt="image" src="https://github.com/user-attachments/assets/2b3e3ff5-b708-4cea-aecb-13c669b0bbe6" />

* There is a lot of variety in the population counts. However, the majority of the municipalities are small, with population counts of less than 10 000.

#### Social and health service expenditures

Let's examine statistical key values of social and health service expenditures per capita of all municipalities:

```python
# Show statistical key values of social and health service expenditures
dfk['Sosiaali- ja terveystoiminta yhteensä, nettokäyttökustannukset, euroa/asukas, 2020'].describe()
```

<img width="594" height="264" alt="image" src="https://github.com/user-attachments/assets/ebc7c566-3a87-447a-b7fa-5fb0af793792" />

* The average social and health service expenditures per capita were 4065.81, the median 4010.60, the minimum 1221.90 and the maximum 6778.70 euros per capita.

Let's create a categorical variable based on social and health service expenditures and add this to the data frame:

```python
# Choose limits for categories based on minimum and maximum values
rajat = [1000, 2000, 3000, 4000, 5000, 6000, 7000]

# Add a new column for social and health service expenditure class to data frame
dfk['st_kust_luok'] = pd.cut(dfk['Sosiaali- ja terveystoiminta yhteensä, nettokäyttökustannukset, euroa/asukas, 2020'], bins=rajat, right=False)

# Create a frequency table for social and health service expenditure class
dfk1 = pd.crosstab(dfk['st_kust_luok'], 'f')
dfk1.columns.name = ''

# Add percentage column and some styling to frequency table
n_kl = dfk1['f'].sum()
dfk1['%'] = dfk1['f']/n_kl*100

dfk1.style.format({'f' : '{:.0f}', '%' : '{:.1f}'})
```

<img width="195" height="207" alt="image" src="https://github.com/user-attachments/assets/f9e26fbb-50b6-419d-abe9-e560d610c208" />

* The most common social and health service expenditure class is 3000 - 4000 euros per capita (39.2%).

Let's create a histogram to visualize this distribution:

```python
sns.histplot(data=dfk['Sosiaali- ja terveystoiminta yhteensä, nettokäyttökustannukset, euroa/asukas, 2020'], bins=rajat)
plt.ylabel(f'Lukumäärä, n = {n_kl}')
plt.xlabel('Sosiaali- ja terveystoiminta, nettokäyttökustannukset, eur/asukas')
plt.show()
```

<img width="492" height="351" alt="image" src="https://github.com/user-attachments/assets/5f640f20-5ad9-42c0-853a-72e7d7106877" />

### Correlations

#### Social and health service expenditures and the percentage of population over 64 years old

Let's examine whether social and health service expenditures correlate with the percentage of population over 64 years old in Finnish municipalities:

```python
# Create a scatterplot
sns.scatterplot(data=dfk, x='Sosiaali- ja terveystoiminta yhteensä, nettokäyttökustannukset, euroa/asukas, 2020', y='Yli 64-vuotiaiden osuus väestöstä, %, 2021')
plt.xlabel('Sosiaali- ja terveystoiminta, nettokäyttökustannukset, eur/asukas')
plt.show()
```

<img width="490" height="350" alt="image" src="https://github.com/user-attachments/assets/4a5b3b9d-b9ad-4e18-8ee5-115f2145b8d7" />

* Based on the shape of the scatterplot it can be anticipated that social and health service expenditures and the percentage of population over 64 years old have a positive correlation.

```python
# Count correlation coefficient
dfk['Sosiaali- ja terveystoiminta yhteensä, nettokäyttökustannukset, euroa/asukas, 2020'].corr(dfk['Yli 64-vuotiaiden osuus väestöstä, %, 2021'])
```

<img width="171" height="30" alt="image" src="https://github.com/user-attachments/assets/e5767691-6304-4fea-b6a6-53e0987a8863" />

* The correlation coefficient of 0.77 confirms a relatively strong positive correlation.

```python
# Use pearsonr test to determine if the correlation is statistically significant
from scipy.stats import pearsonr
pearsonr(dfk['Sosiaali- ja terveystoiminta yhteensä, nettokäyttökustannukset, euroa/asukas, 2020'], dfk['Yli 64-vuotiaiden osuus väestöstä, %, 2021'])
```

<img width="505" height="29" alt="image" src="https://github.com/user-attachments/assets/9ee8fd78-c2e3-4fc8-a6dc-decc506099e5" />

* According to PearsonR test social and health service expenditures and the percentage of population over 64 years old have a positive correlation (r=0.77) and this correlation is statistically significant (p<0.01).

#### Percentage of Swedish-speaking population and education and culture expenditures

Let's examine whether the percentage of Swedish-speaking population correlates with education and culture expenditures per capita in Finnish municipalities:

```python
# Create a scatterplot
sns.scatterplot(data=dfk, x='Opetus- ja kulttuuritoiminta yhteensä, nettokäyttökustannukset, euroa/asukas, 2020', y='Ruotsinkielisten osuus väestöstä, %, 2021')
plt.xlabel('Opetus- ja kulttuuritoiminta, nettokäyttökustannukset, eur/asukas')
plt.show()
```

<img width="484" height="351" alt="image" src="https://github.com/user-attachments/assets/7070ed4e-cc28-4865-af0d-6fd220a83c1c" />

* Based on the scatterplot, it is difficult to establish a clear correlation.

```python
# Count correlation coefficient
dfk['Opetus- ja kulttuuritoiminta yhteensä, nettokäyttökustannukset, euroa/asukas, 2020'].corr(dfk['Ruotsinkielisten osuus väestöstä, %, 2021'])
```

<img width="151" height="27" alt="image" src="https://github.com/user-attachments/assets/e2df2df6-56a1-4db3-95a7-73358bbe562b" />

* The correlation coefficient shows a weak positive correlation (r=0.35)

```python
# Use pearsonr test to determine if the correlation is statistically significant
pearsonr(dfk['Opetus- ja kulttuuritoiminta yhteensä, nettokäyttökustannukset, euroa/asukas, 2020'], dfk['Ruotsinkielisten osuus väestöstä, %, 2021'])
```

<img width="519" height="27" alt="image" src="https://github.com/user-attachments/assets/62219401-6e66-488c-aeea-bb03135ccc38" />

* According to PearsonR test the weak positive correlation (r=0.35) is statistically significant (p<0.01).

