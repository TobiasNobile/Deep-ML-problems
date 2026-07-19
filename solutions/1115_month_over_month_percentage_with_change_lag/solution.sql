WITH sum_month AS (
    SELECT 
    date_trunc('month', sale_date) AS month, 
    SUM(amount) AS total, 
    LAG(SUM(amount)) OVER (ORDER BY date_trunc('month', sale_date)) AS previous
    FROM sales
    GROUP BY date_trunc('month', sale_date)
)

SELECT month, total, ROUND((total - previous) / previous * 100, 2) AS pct_change
FROM sum_month;