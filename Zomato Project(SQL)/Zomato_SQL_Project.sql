create database zomato
Use zomato

drop table if exists goldusers_signup;
CREATE TABLE goldusers_signup(userid integer,gold_signup_date date); 

INSERT INTO goldusers_signup(userid, gold_signup_date) 
VALUES 
(1, '2017-09-22'),
(3, '2017-04-21');

drop table if exists users;
CREATE TABLE users(userid integer,signup_date date); 

INSERT INTO users(userid, signup_date) 
VALUES 
(1, '2014-09-02'),
(2, '2015-01-15'),
(3, '2014-04-11');

drop table if exists sales;
CREATE TABLE sales(userid integer,created_date date,product_id integer); 

INSERT INTO sales(userid, created_date, product_id) 
VALUES 
(1, '2017-04-19', 2),
(3, '2019-12-18', 1),
(2, '2020-07-20', 3),
(1, '2019-10-23', 2),
(1, '2018-03-19', 3),
(3, '2016-12-20', 2),
(1, '2016-11-09', 1),
(1, '2016-05-20', 3),
(2, '2017-09-24', 1),
(1, '2017-03-11', 2),
(1, '2016-03-11', 1),
(3, '2016-11-10', 1),
(3, '2017-12-07', 2),
(3, '2016-12-15', 2),
(2, '2017-11-08', 2),
(2, '2018-09-10', 3);



drop table if exists product;
CREATE TABLE product(product_id integer,product_name text,price integer); 

INSERT INTO product(product_id,product_name,price) 
 VALUES
(1,'p1',980),
(2,'p2',870),
(3,'p3',330);


select * from sales;
select * from product;
select * from goldusers_signup;
select * from users;


# 1.what is total amount each customer spent on zomato ?
Select a.userid, sum(b.price) as total_spent
from sales as a
join product as b
on a.product_id=b.product_id
group by a.userid


#2.How many days has each customer visited zomato?
Select userid, count(Distinct created_date) No_days_visited
from sales
group by userid


#3.what was the first product purchased by each customer?
Select userid, product_id, created_date
from Sales where (userid, created_date) in
(
Select userid, min(created_date) first_day
from sales
group by userid) 

select userid, product_id, created_date as first_day from 
(Select userid, product_id, created_date, rank() over(partition by userid order by created_date) as rnk
from sales
)as x 
where rnk=1

With CTE as 
(Select userid, product_id, created_date, rank() over(partition by userid order by created_date) as rnk
from sales
)select userid, product_id, created_date as first_day from CTE
where rnk=1

# 4.what is most purchased item on menu & how many times was it purchased by all customers ?
Select product_id, count(product_id) as total_sold
from sales
group by product_id
order by count(product_id) DESC
limit 1


WITH ranked_sales AS (
    SELECT product_id, COUNT(product_id) AS total_sold,
           ROW_NUMBER() OVER (ORDER BY COUNT(product_id) DESC) AS rnk
    FROM sales
    GROUP BY product_id
)
SELECT product_id, total_sold
FROM ranked_sales
WHERE rnk = 1;




#5.which item was most popular for each customer?
With CTE as (Select userid, product_id, rank() over(partition by userid order by count(product_id) DESC) as rnk
from sales
group by userid, product_id
order by userid, count(product_id) DESC)
select userid, product_id
from CTE
where rnk=1


#6.which item was purchased first by customer after they become a member ?
with CTE as (select a.userid, a.product_id, rank() over(partition by a.userid order by a.created_date ASC) as rnk, gold_signup_date, 
created_date from sales as a
join goldusers_signup as b
on a.userid=b.userid
where gold_signup_date<created_date) Select userid,product_id, created_date, gold_signup_date from CTE
where rnk=1



#7. which item was purchased just before the customer became a member?
with CTE as (select a.userid, a.product_id, rank() over(partition by a.userid order by a.created_date DESC) as rnk, gold_signup_date, 
created_date from sales as a
join goldusers_signup as b
on a.userid=b.userid
where gold_signup_date>created_date) Select userid,product_id, created_date, gold_signup_date from CTE
where rnk=1

select * from sales
select * from product
select * from goldusers_signup

#8. what is total orders and amount spent for each member before they become a member?
select a.userid, count(a.created_date) as Prev_count_order, sum(c.price)
created_date from sales as a
join goldusers_signup as b
join product as c
on a.userid=b.userid
and a.product_id=c.product_id
where gold_signup_date>created_date
group by a.userid



#9. If buying each product generates points for eg 5rs=2 zomato point 
#and each product has different purchasing points for eg for p1 5rs=1 zomato point,
#for p2 10rs=1 zomato point and p3 20rs=1 zomato point  2rs =1zomato point, 
#calculate points collected by each customer and for which product most points have been given till now.

With CTE as (Select a.userid, a.product_id, sum(b.price) as total_spent,
case when a.product_id=1 then 2*(sum(b.price)/5) 
when a.product_id=2 then sum(b.price)/10
when a.product_id=3 then sum(b.price)/20 end as Zomato_points, rank() over(partition by userid order by case when a.product_id=1 then 2*(sum(b.price)/5) 
when a.product_id=2 then sum(b.price)/10
when a.product_id=3 then sum(b.price)/20 end DESC) as rnk
from sales as a
Join product as b
on a.product_id= b.product_id
group by a.userid, a.product_id
order by a.userid)
Select userid, product_id, Zomato_points
from CTE
where rnk=1




#10. In the first year after a customer joins the gold program (including the join date ) 
#irrespective of what customer has purchased earn 5 zomato points for every 10rs spent who earned more more 1 or 3 what int earning in first yr ? 1zp = 2rs

With CTE as
(Select a.userid, a.product_id, sum(b.price) as total_spent,
case when a.product_id=1 then 2*(sum(b.price)/5) 
when a.product_id=2 then sum(b.price)/10
when a.product_id=3 then sum(b.price)/20 end as Zomato_points, 
rank() over(partition by userid order by case when a.product_id=1 then 2*(sum(b.price)/5) 
when a.product_id=2 then sum(b.price)/10
when a.product_id=3 then sum(b.price)/20 end DESC) as rnk
from sales as a
Join product as b
join goldusers_signup as c
on a.product_id= b.product_id 
and a.userid=c.userid
where datediff(c.gold_signup_date, a.created_date)<=365
group by a.userid, a.product_id
order by a.userid) 
select userid, round(sum(Zomato_points),2) as zmt_pnts, round(sum(Zomato_points)*2,2) as Rupees_value
from CTE
group by userid



#11. rnk all transaction of the customers
Select userid, count(created_date) no_of_tansactions, Dense_rank() over(order by count(created_date) DESC) as rnk
from sales
group by userid





#12. rank all transaction for each member whenever they are zomato gold member for every non gold member transaction mark as na
Select a.userid, count(a.created_date) no_of_tansactions, 
case when a.userid=b.userid then Dense_rank() over(order by count(created_date) DESC) else "na" end as rnk
from sales as a
left join goldusers_signup as b
on a.userid=b.userid
group by a.userid










