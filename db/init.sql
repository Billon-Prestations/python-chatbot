-- SQLite
DROP TABLE IF EXISTS patterns;
DROP TABLE IF EXISTS tags;

CREATE TABLE tags (
    idtag INTEGER PRIMARY KEY AUTOINCREMENT,
    libelletag txt
);

INSERT INTO tags 
(idtag, libelletag)
VALUES 
(1, 'greeting'),
(2, 'goodbye'),
(3, 'thanks'),
(4, 'whoareyou'),
(5, 'name'),
(6, 'help'),
(7, 'createaccount'),
(8, 'time'),
(9, 'complaint'),
(10, 'speak');

SELECT * FROM tags;

CREATE TABLE patterns (
    idpattern INTEGER PRIMARY KEY AUTOINCREMENT,
    idtag INTEGER,
    libellepattern txt
);

INSERT INTO patterns
(idtag, libellepattern)
VALUES
(1, 'Hi'),
(1, 'Hey'),
(1, 'Is anyone there?'),
(1, 'Hello'),
(1, 'Hola'),
(1, 'Salut'),
(1, 'Wesh alors'),
(1, 'Bonjour');

SELECT * FROM patterns;