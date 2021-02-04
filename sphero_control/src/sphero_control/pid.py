class PID(object):
    def __init__(self, kp, ki, kd, Td, lim_lo, lim_hi):
        # Initialize gains.
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain

        # Initialize limits.
        self.lim_hi = lim_hi  # Upper limit
        self.lim_lo = lim_lo  # Lower limit

        # Initialize control values.
        self.error_old = 0
        self.ui_old = 0

        # Initialize other values.
        self.Td = Td  # Sampling period

    def compute(self, ref, meas):
        """
        Perform a PID computation and return a control value based on
        the elapsed time (dt) and the error signal.

        :param ref: referent value
        :param meas: measured value
        :return: control value
        """

        error = ref - meas

        # Compute P, I and D parts
        up = self.kp * error
        ui = self.ki * error * self.Td + self.ui_old
        ud = self.kd * (error - self.error_old) / self.Td

        # Compute total control value
        u = up + ui + ud

        # Saturation and anti wind-up
        if u > self.lim_hi:
            u = self.lim_hi
            ui = self.ui_old
        elif u < self.lim_lo:
            u = self.lim_lo
            ui = self.ui_old

        if abs(error) < 5:
            u = 0


        # Save current values for next iteration
        self.ui_old = ui
        self.error_old = error

        return u
