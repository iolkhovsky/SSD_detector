from os.path import isfile


class LRBehavior:
    LR_CONSTANT = 0
    LR_LINEAR = 1


def load_training_schedule(batch_per_epoch, path="lr_schedule.txt", logger=None):
    # defres = [{"value_start": def_lr, "value_stop": def_lr, "type": LRBehavior.LR_CONSTANT, "duration": 1}]
    if isfile(path):
        plan = []
        with open(path, "r") as f:
            line = f.readline()
            idx = 0
            while line:
                if idx > 0:
                    # not a header
                    wds = line.split()
                    if len(wds) == 5:
                        stage = dict()
                        stage["value_start"] = float(wds[0])
                        stage["value_stop"] = float(wds[1])
                        stage_beh = None
                        if wds[2] == "linear":
                            stage_beh = LRBehavior.LR_LINEAR
                        if wds[2] == "const":
                            stage_beh = LRBehavior.LR_CONSTANT
                        if stage_beh is None:
                            if logger:
                                logger("Stage behavior undefined!")
                            raise RuntimeError("Error while parsing lr schedule file")
                        stage["type"] = stage_beh
                        if wds[4] == "epochs":
                            stage["duration"] = float(wds[3]) * batch_per_epoch
                        else:
                            stage["duration"] = float(wds[3])
                        plan.append(stage)
                    else:                       # some kind of error
                        if logger:
                            logger("Unexpected string lenghtin lr schedule ini file!")
                        raise RuntimeError("Error while parsing lr schedule file")
                line = f.readline()
                idx += 1
    else:
        if logger:
            logger("File with training schedule is invalid!")
        raise RuntimeError("LR schedule file is invalid")
    return plan


class LRScheduler:

    def __init__(self, schedule=[]):
        if not isinstance(schedule, list):
            raise ValueError("LRScheduler must recieve list of {'value_start', 'value_stop', 'type', 'duration'}")
        self.current_lr = 1e-4
        self.current_step_id = 0
        self.current_stage = 0
        self.local_step = 0

        self.schedule = schedule

        self.stage_switching = [self.current_step_id]
        for i in range(len(self.schedule)-1):
            prev = self.stage_switching[i]
            self.stage_switching.append(prev + int(self.schedule[i]["duration"]))
        self.stage_switching = self.stage_switching[1:]
        pass

    def step(self):
        self.current_step_id += 1
        self.local_step += 1
        if self.need_update():
            self.current_stage += 1
            self.local_step = 0
        pass

    def set_position(self, start_pos):
        self.current_step_id = start_pos
        pass

    def need_update(self):
        return self.current_step_id in self.stage_switching

    def get_pars(self):
        if len(self.schedule):
            cur_stage_schedule = self.schedule[self.current_stage]
            if self.local_step < int(cur_stage_schedule["duration"]):
                if cur_stage_schedule["type"] == LRBehavior.LR_CONSTANT:
                    self.current_lr = cur_stage_schedule["value_start"]
                elif cur_stage_schedule["type"] == LRBehavior.LR_LINEAR:
                    start = cur_stage_schedule["value_start"]
                    stop = cur_stage_schedule["value_stop"]
                    dur = int(cur_stage_schedule["duration"])
                    self.current_lr = start + (stop - start) * self.local_step / dur
        return self.current_lr, self.current_step_id

    def get_scheduled_duration(self):
        out = 0
        for stage in self.schedule:
            out += int(stage["duration"])
        return out

    def __str__(self):
        return "LR scheduler with sets: " + str(self.schedule)


class OptimizerWithSchedule:

    def __init__(self, optim, scheduler=LRScheduler()):
        self.optimizer = optim
        self.scheduler = scheduler
        self.lr, _ = scheduler.get_pars()
        pass

    def zero_grad(self):
        self.optimizer.zero_grad()
        pass

    def step(self):
        self.optimizer.step()
        self.scheduler.step()
        if self.scheduler.need_update():
            self.lr, _ = self.scheduler.get_pars()
            self.__refresh_lr__()
        pass

    def __refresh_lr__(self):
        for g in self.optimizer.param_groups:
            g['lr'] = self.lr
        pass


def load_default_schedule_plan(batch_per_epoch, path="lr_schedule.txt", def_lr=1e-3):
    defres = [{"value_start": def_lr, "value_stop": def_lr, "type": LRBehavior.LR_CONSTANT, "duration": 1}]
    defpath = path
    if isfile(defpath):
        plan = []
        with open(defpath, "r") as f:
            line = f.readline()
            idx = 0
            while line:
                if idx > 0:
                    # not a header
                    wds = line.split()
                    if len(wds) == 5:
                        stage = dict()
                        stage["value_start"] = float(wds[0])
                        stage["value_stop"] = float(wds[1])
                        stage_beh = None
                        if wds[2] == "linear":
                            stage_beh = LRBehavior.LR_LINEAR
                        if wds[2] == "const":
                            stage_beh = LRBehavior.LR_CONSTANT
                        if stage_beh is None:
                            return defres
                        stage["type"] = stage_beh
                        if wds[4] == "epochs":
                            stage["duration"] = float(wds[3]) * batch_per_epoch
                        else:
                            stage["duration"] = float(wds[3])
                        plan.append(stage)
                    else:                       # some kind of error
                        return defres
                line = f.readline()
                idx += 1
    else:
        return defres
    return plan

# schedule_plan = [
#     {"value_start": 1e-3, "value_stop": 1e-2, "type": LRBehavior.LR_LINEAR, "duration": 10 * batches_per_epochs},
#     {"value_start": 1e-2, "value_stop": 1e-2, "type": LRBehavior.LR_CONSTANT, "duration": 75 * batches_per_epochs},
#     {"value_start": 1e-2, "value_stop": 1e-3, "type": LRBehavior.LR_LINEAR, "duration": 30 * batches_per_epochs},
#     {"value_start": 1e-3, "value_stop": 1e-4, "type": LRBehavior.LR_LINEAR, "duration": 30 * batches_per_epochs}]
