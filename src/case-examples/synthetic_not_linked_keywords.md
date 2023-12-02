# Title:
[AUTO] Тест без ссылок на кейворды

# Name in TORS:
library/application/synthetic/not_linked_keywords.py

# ID:
131

# URL:
https://some_url/131

# Section ID:
13

# Refs:


# Main Product:
Synthetic

# Description:
С помощью полученных логина и пароля для системы NNN выполнить запрос 
к системе и убедиться, что параметр PPP отображается корректно.

# Preconditions:

Получены логин и пароль для входа в систему NNN.
Имеется достаточный уровень прав для выполнения запроса.

# STEP №1

## Content:
Написать запрос вида `LOGIN: <login>, PASSWORD: <password>` с данными, 
которые получили перед прохождением кейса.

## Expected:
Параметр PPP после запроса отображается корректно.