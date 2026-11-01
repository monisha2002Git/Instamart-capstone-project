-- Part A, task 6 -- seven analytical queries against data/instamart.db.
-- Unless stated otherwise, only completed orders count as revenue or spend.
-- Each query is preceded by a "-- name: Qn" marker so src/a4_sql.py can run them individually.

-- name: Q1
-- WHERE then HAVING: keep completed orders over 500 (WHERE), group by customer,
-- keep customers whose total of those orders exceeds 8000 (HAVING).
SELECT customer_id,
       ROUND(SUM(order_amount), 2) AS total_over_500
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_over_500 DESC;

-- name: Q2
-- Three-table join: revenue by city and product category.
SELECT c.city,
       p.category,
       ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products  p ON o.product_id  = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY revenue DESC;

-- name: Q3
-- Scalar subquery: products priced above the average catalogue price.
SELECT product_id, product_name, category, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- name: Q4
-- LEFT JOIN: customers who have no completed order.
SELECT c.customer_id, c.city
FROM customers c
LEFT JOIN orders o
       ON c.customer_id = o.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- name: Q5
-- CTE: customers whose total completed spend is above the average of those totals.
WITH spend AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
avg_spend AS (
    SELECT AVG(total_spend) AS mean_total FROM spend
)
SELECT s.customer_id, ROUND(s.total_spend, 2) AS total_spend
FROM spend s
CROSS JOIN avg_spend a
WHERE s.total_spend > a.mean_total
ORDER BY s.total_spend DESC;

-- name: Q6
-- Ranking window functions over completed spend, partitioned by city.
WITH spend AS (
    SELECT o.customer_id, c.city, SUM(o.order_amount) AS total_spend
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY o.customer_id, c.city
)
SELECT customer_id,
       city,
       ROUND(total_spend, 2) AS total_spend,
       RANK()       OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
FROM spend
ORDER BY city, total_spend DESC;

-- name: Q7
-- Running total: daily completed revenue with a cumulative column.
WITH daily AS (
    SELECT substr(order_ts, 1, 10) AS day,
           SUM(order_amount)       AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY substr(order_ts, 1, 10)
)
SELECT day,
       ROUND(revenue, 2)                                      AS revenue,
       ROUND(SUM(revenue) OVER (ORDER BY day), 2)             AS cumulative_revenue
FROM daily
ORDER BY day;
