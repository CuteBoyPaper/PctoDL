import math
class PID:
    def __init__(self,power_cap,alpha,kp,ki,kd,index_list,log_txt):
        #功率上限约束
        self.power_cap = power_cap 
        self.alpha = 100 * (1-alpha)
        #当前功率
        self.current_power = 0
        #误差值
        self.error = 0
        #系数
        self.kp = kp #比例
        self.ki = ki #积分
        self.kd = kd #微分
        self.cur_kp = kp
        self.cur_ki = ki
        self.cur_kd = kd
        self.previous_error = 0
        self.last_error = 0
        self.times=0
        self.integral = 0
        self.index_list = index_list
        self.log_txt = log_txt
    def scheduler(self,current_power,current):
        self.error = self.power_cap - current_power
        delta_t = 100*self.error / self.power_cap
        if delta_t <= self.alpha and delta_t > -1:
            self.times=0
            self.integral=0
            return current
        self.integral += self.error
        differential = self.error - 2*self.previous_error + self.last_error
        # 若误差正负号改变, 则重置积分项, 从而保证积分项对于最近累计/稳态误差的快速反应/响应
        if self.error*self.previous_error<0 :
            self.integral = self.error
            self.cur_ki = 0
        elif self.times!=0:
            self.cur_ki=self.ki*math.pow(100,self.times)
        self.log_txt.write("error: {:.6f}\t power cap: {:6f}\t bili: {:.6f} times:{}\n".format(self.error,self.power_cap,delta_t,self.times))
        delta_t = abs(delta_t)
        if delta_t > 50:
            self.cur_kp = self.kp * 50
            # self.log_txt.write("1111111111\n")
        elif delta_t>30 and delta_t<=50:
            self.cur_kp = self.kp * 30
            # self.log_txt.write("2222222222\n")
        elif delta_t>10 and delta_t<=30:
            self.cur_kp = self.kp * 20
        elif delta_t>5 and delta_t<=10:
            self.cur_kp = self.kp 
        elif delta_t >=0:
            self.cur_kp = self.kp 
        # self.cur_ki = self.ki
        # self.cur_kp = self.kp
        self.cur_kd = self.kd
        delta = self.cur_kp * self.error + self.cur_ki * self.integral + self.cur_kd * differential
        self.log_txt.write("cur_kp: {:.6f}\t cur_ki: {:6f}\t cur_kd: {:.6f}\n".format(self.cur_kp,self.cur_ki,self.cur_kd))
        self.log_txt.write("error: {:.3f}\t integral: {:3f}\t differential: {:.3f}\t delta: {:.3f}\n".format(self.error,self.integral,differential,delta))
        next_index = 0
        if self.index_list.index(current)<=1:
            next_index = math.ceil((self.index_list.index(current)+1) * (1 + delta))-1    
        else:
            next_index = math.floor((self.index_list.index(current)) * (1 + delta))
        self.log_txt.write("current index: {}\t next index: {}\n ".format(self.index_list.index(current),next_index))
        next_index = max(next_index,0)
        next_index = min(next_index,len(self.index_list)-1)    
        self.log_txt.write("current : {:.3f}\t index: {}\t next index: {}\t next: {:3f}\n ".format(current,self.index_list.index(current),next_index,self.index_list[next_index])) 
        self.last_error = self.previous_error
        self.previous_error = self.error
        if next_index==self.index_list.index(current) and delta_t> 5:
            self.times+=1
        else:
            self.times=0
        return self.index_list[next_index]
 
class PIDScheduler:
    def __init__(self,sm_pid,mem_pid,bs_pid,log_txt):
        self.sm_pid = sm_pid
        self.mem_pid = mem_pid
        self.bs_pid = bs_pid
        self.log_txt = log_txt
    def scheduler(self,current_power,current_sm,current_mem,current_bs):
        self.log_txt.write("sm info\n")
        next_sm = self.sm_pid.scheduler(current_power,current_sm)
        next_mem=current_mem
        next_bs=current_bs
        # self.log_txt.write("mem info\n")
        # next_mem = self.mem_pid.scheduler(current_power,current_mem)
        # self.log_txt.write("bs info\n")
        # next_bs = self.bs_pid.scheduler(current_power,current_bs)
        return next_sm,next_mem,next_bs
class Morak:
    def __init__(self,power_cap,alpha,slo,belta,sm_clocks,batch_size_list,log_txt):
        self.power_cap = power_cap
        self.alpha = alpha
        self.belta = belta
        self.slo = slo
        self.sm_clocks = sm_clocks
        self.batch_size_list = batch_size_list
        self.log_txt = log_txt
        
    def scheduler(self,lantecy,max_power,frequency,batch_size):
        sm_index = self.sm_clocks.index(frequency)
        batch_index = self.batch_size_list.index(batch_size)
        #对比算法morak算法的实现
        if max_power<=self.alpha*self.power_cap:
            # if lantecy<=self.slo and lantecy>self.belta*self.slo :
            #     sm_index+=1
            # elif lantecy<=self.belta*self.slo:
            #     batch_index+=1
            # else:
            #     batch_index-=1
            if lantecy<=self.belta*self.slo:
                batch_index+=1
            else:
                sm_index+=1
        elif max_power>self.alpha*self.power_cap and max_power<=1.02*self.power_cap:
            pass
        else:
            if lantecy>self.belta*self.slo:
                batch_index-=1
            elif lantecy<=self.belta*self.slo:
                sm_index-=1
        sm_index = min(max(sm_index,0),len(self.sm_clocks)-1)
        batch_index = min(max(batch_index,0),len(self.batch_size_list)-1)
        self.log_txt.write("next sm index: {}\t next sm: {:3f}\n ".format(sm_index,self.sm_clocks[sm_index])) 
        self.log_txt.write("next batch index: {}\t next batchsize: {:3f}\n ".format(batch_index,self.batch_size_list[batch_index])) 
        return self.sm_clocks[sm_index],self.batch_size_list[batch_index]


class BatchDVFS:
    def __init__(self,power_cap,alpha,sm_clocks,batch_size_list,log_txt):
        self.power_cap = power_cap
        self.alpha = alpha
        self.min_bs = 0
        self.max_bs = len(batch_size_list)
        self.sm_clocks = sm_clocks
        self.batch_size_list = batch_size_list
        self.log_txt = log_txt
    def scheduler(self,max_power,frequency,batch_size):
        sm_index = self.sm_clocks.index(frequency)
        batch_index = self.batch_size_list.index(batch_size)
        if max_power<=self.power_cap and max_power> self.alpha*self.power_cap:
            pass
        elif max_power<=self.power_cap*self.alpha:
            self.min_bs = batch_index
            batch_index =math.ceil((self.min_bs+self.max_bs)/2)
            if batch_index == len(self.batch_size_list):
                sm_index +=5
        else:
            if batch_index==0:
                self.max_bs = batch_index
                self.min_bs = 0
                batch_index = math.floor((self.min_bs+self.max_bs)/2)
            else:
                self.max_bs = batch_index
                batch_index = math.floor((self.min_bs+self.max_bs)/2)
            if batch_index ==0:
                sm_index -=3
        sm_index = min(max(sm_index,0),len(self.sm_clocks)-1)
        batch_index = min(max(batch_index,0),len(self.batch_size_list)-1)
        self.log_txt.write("next sm index: {}\t next sm: {:3f}\n ".format(sm_index,self.sm_clocks[sm_index])) 
        self.log_txt.write("next batch index: {}\t next batchsize: {:3f}\n ".format(batch_index,self.batch_size_list[batch_index])) 
        return self.sm_clocks[sm_index],self.batch_size_list[batch_index]
        