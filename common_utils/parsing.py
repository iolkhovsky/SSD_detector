

def parse_period_string(period):
    if period:
        postfix = period[-1]
        period = period[:-1]
        value = int(period)
        if postfix == "e":
            return value, None
        elif postfix == "b":
            return None, value
        else:
            raise ValueError("Wrong format of period")
    else:
        return None, None
