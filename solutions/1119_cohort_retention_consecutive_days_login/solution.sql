SELECT DISTINCT user_id, MIN(login_date) AS first_login
FROM logins
WHERE user_id IN (
    SELECT l1.user_id
    FROM logins l1
    JOIN logins l2 ON l1.login_date = l2.login_date - 1
    AND l1.user_id = l2.user_id
)

GROUP BY user_id
ORDER BY user_id;
