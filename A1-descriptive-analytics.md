# Python data analytics - Assignment 1 - descriptive analytics

Author: Toni Blom

Created: 2025-09-15

Original [Google Colab file](https://colab.research.google.com/drive/1Vzr6uneZcaf0LvN0MVnRhBa1fgQoOau7?usp=sharing) (in Finnish).

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

## Data set 1: Titanic

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

Let's try to filter the data:

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







