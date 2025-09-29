---
layout: post
title:  "Tuning a PID Controller"
categories: jekyll update
---


If you’ve ever tried to control the speed of a motor, balance a robot, or maintain a 
stable temperature with an Arduino or any other micro-controller, chances are you’ve 
come across something called a **PID controller**. At first, PID might look complicated 
with all its math and jargon—but its actually a simple concept! Today, I will 
break it down for you in simple terms so that you can quickly learn how to implement 
it in your own projects.  


---


## What is a PID Controller?

PID stands for:  
- **P**: Proportional  
- **I**: Integral  
- **D**: Derivative  

Think of it like this:  

- **P** is how hard you push toward your goal.
- **I** keeps track of how far you’ve been off the goal over time and nudges you to 
fix it.
- **D** watches how fast you’re approaching the goal and prevents you from overshooting.

All three work together to reach your target smoothly and accurately.


---


## Implementing PID Controller in Code

Writing the PID logic in code has to be the easiest part of all. There are only few 
thing to keep in mind.

- Constant main loop

```c
unsigned long loop_timer;

loop_timer = micros();

void loop() {

  /* Read sensor value */

  /* PID code */

  while (micros() - loop_timer < 4000);
  loop_timer = micros();
}
```

- Calculate error

```c
float error, prev_error;
error = set_point - sensor_value;
```

- Calculate **p** value

```c
float p;
p = kp * error;
```

- Calculate **d** value

```c
float d;
d = kd * (prev_error - error);
```

- Calculate **i** value

```c
float i;
i = i + ki * error;
```

- Total PID value

```c
float output;
output = p + i - d;
```

---

## Tuning the PID Constants

Now all you have to do is change the values of `kp`, `ki` and `kd` such that you get 
desired output from your system. 

1. Start with only one constant `kp`, like `kp = 2.0, ki = 0, kd = 0` and observe the 
output. Tweak this value so that you get optimum possible outcome.

2. Next, start assuming `kd,` yeah `kd` not `ki`. We secondly tune `kd` because, it 
dampens the over and under shoot created by `kp`. You might need to go back to `kp` 
and tweak it to tune this value properly.

3. Finally, tune the `ki`. For the many systems you might not even need `ki`. You have 
to think slightly differently for constant `ki` because if you see above mathematical 
expression for the value associated with `ki` which is—`i`, depends on previous values 
of `i` meaning it is integrating all the values over many loops whereas for `d` you 
only need what previous `error` value was, meaning it is accounting for how large the 
difference in `error` is between the loops and finally for `kp`, it just needs the 
current `error`.


---


## Other Important Points

1. Calibrating the sensor—You might need to add offset values depending upon the type 
and quality of sensor used in the project.

2. Skip the processing in deadzone—As we know the total PID value is the direct output 
to the actuator, suppose, for a balancing robot in a *single axis*, if the sensor 
output value in degree is $$ 0^\circ $$ when perfectly vertical, then possible deadzone 
range (depending upon the type of sensors, motors etc.) can be $$ \pm2^\circ $$, where 
processing the PID value can be completely skipped and simply set to 0.

3. Also Keep in account for the actuator deadzone—If you are using an arduino to drive 
a motor with a motor driver, you have to give out PWM value from any analog pin to the 
driver, which generally in code value from 0 to 255 and you might have noticed the 
motor does not immediately start moving in initial values maybe even for upto 10 or 20, 
this maximum value until when the motor does not start moving is the actuator deadzone. 
And to fix this problem simply add this value 20 to PID output or you can also map your 
PID value to range 20 to 255 of motor driver output.

4. Clamp the PID value to your actuator range—Suppose if your total PID value came out 
to be 257 to give it to your motor driver which should be only between range 0 to 255, 
the value might overflow as 2. So better clamp the value $$ \gt $$ 255 to 255 and 
$$ \lt $$ 0 to 0.


---


## Combining Everything

A typical setup looks like this:

```cpp
#include <Arduino.h>

/* Required variables */
float sensor_value, output, set_point, error, prev_error, p, i, d;
unsigned long loop_timer;

/* PID parameters */
float Kp = 2.0, Ki = 5.0, Kd = 1.0;

/* Utility functions */
float read_sensor() {}
void drive_motor(float value) {}

void setup() {

  /* Required setup code ... */
  
  loop_timer = micros();
}

void loop() {

  /* Read sensor value */
  sensor_value = read_sensor();

  /* PID code */
  error = set_point - sensor_value;
  p = kp * error;
  d = kd * (prev_error - error);
  prev_error = error;
  i = i + ki * error;
  output = p + i - d;

  /* Driving the actuator */
  drive_motor(output);

  /* Constant 250Hz main loop */
  while (micros() - loop_timer < 4000);
  loop_timer = micros();
}
```