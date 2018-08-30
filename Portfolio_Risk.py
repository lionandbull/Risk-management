import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
from scipy.stats import t
import pickle
import statsmodels.tsa.api as smt
from arch import arch_model
import seaborn as sns
from prettytable import PrettyTable

class Portfolio_Risk:
    def __init__(self, stock_amount, *files):
        self.L_act = []
        self.stock_amount = [stock_amount]
        self.files = files
        self.price_list = []
        self.log_return = []
        self.V_t = []
        self.date_list = []
        self.monthly_end_date = []
        self.shares = []
        self.monthly_index = []
        self.lambda_dict = []
        self.lambda_ = []
        self.mu = []
        self.cov_matrix = []
        self.weekly_index = []
        self.weekly_date = []
        self.bond_weekly_date = []
        self.bond_weekly_index = []
        self.bond_date = []
        self.bond_price = []
        self.bond_yield = []
        self.bond_lambda = []
        self.option_date = []
        self.option_price = []
        self.option_greeks = []
        self.option_iv = []
        self.option_ir = []
        self.option_ir_date = []
        self.option_weekly_date = []
        self.option_weekly_index = []
        self.XEO_price = []
        self.XEO_log_return = []

    def get_price_list(self):
        # Get whole price list and date list
        date_list = []
        price_list = []
        for file in self.files:
            if file == 'XEO_price.csv':
                csv = pd.read_csv(file)
                list_0 = []
                for item in csv.as_matrix()[6:][::-1]:
                    list_0.append(float(item[1]))
                self.XEO_price.append(np.asarray(list_0))
            else:
                csv = pd.read_csv(file)
                csv.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
                list_0 = []
                list_1 = []
                for item in csv.as_matrix():
                    list_0.append(item[0])
                    list_1.append(item[1])
                date_list.append(np.asarray(list_0))
                price_list.append(np.asarray(list_1))
        self.price_list = np.asarray(price_list)
        self.XEO_price = np.asarray(self.XEO_price)
        date_list = np.asarray(date_list)
        for i in range(len(date_list)):
            for j in range(len(date_list[i])):
                date_list[i][j] = datetime.datetime.strptime(date_list[i][j], '%Y-%m-%d').strftime('%-m/%-d/%y')
        self.date_list = np.asarray(date_list)

        for i in price_list:
            returns = []
            for j in range(len(i) - 1):
                returns.append((i[j + 1] - i[j]) / i[j])
            self.log_return.append(np.log(1 + np.asarray(returns)))
        self.log_return = np.asarray(self.log_return)

        for i in self.XEO_price:
            returns = []
            for j in range(len(i) - 1):
                returns.append((i[j + 1] - i[j]) / i[j])
            self.XEO_log_return.append(np.log(1 + np.asarray(returns)))
        self.XEO_log_return = np.asarray(self.XEO_log_return)
        self.XEO_price = self.XEO_price[0][1:]
        self.XEO_log_return = self.XEO_log_return[0]
        for i in range(len(self.price_list)):
            index = np.nonzero(self.date_list[i] == '1/4/16')[0][0]
            self.price_list[i] = self.price_list[i][index:-1]
            self.date_list[i] = self.date_list[i][index:-1]
            self.log_return[i] = self.log_return[i][index - 1:-1]

    def get_bond_list(self, *files):
        for file in files:
            csv = pd.read_csv(file)
            list_0 = []
            list_1 = []
            list_2 = []
            for item in csv.as_matrix()[6:][::-1]:
                list_0.append(item[0])
                list_1.append(float(item[1]))
                list_2.append(float(item[2])*0.01)
            self.bond_date.append(np.asarray(list_0))
            self.bond_price.append(np.asarray(list_1))
            self.bond_yield.append(np.asarray(list_2))
        self.bond_date = np.asarray(self.bond_date)
        self.bond_price = np.asarray(self.bond_price)
        self.bond_yield = np.asarray(self.bond_yield)
        self.bond_date[0] = self.bond_date[0][330:]
        self.bond_price[0] = self.bond_price[0][330:]
        self.bond_yield[0] = self.bond_yield[0][330:]

    def get_option_list(self, *files):
        for file in files:
            if file == 'Greek.csv':
                csv = pd.read_csv(file)
                list_all = []
                for item in csv.as_matrix()[1:10][::-1]:
                    list_all.append([float(item[4]), float(item[1])*0.01, float(item[5]), float(item[3])])
                self.option_greeks.append(list_all)

                list_all = []
                for item in csv.as_matrix()[13:22][::-1]:
                    list_all.append([float(item[4]), float(item[1]) * 0.01, float(item[5]), float(item[3])])
                self.option_greeks.append(list_all)

                list_all = []
                for item in csv.as_matrix()[25:34][::-1]:
                    list_all.append([float(item[4]), float(item[1]) * 0.01, float(item[5]), float(item[3])])
                self.option_greeks.append(list_all)
            elif file == 'KNX_vol.csv' or file == 'NFLX_vol.csv' or file == 'XEO_vol.csv':
                csv = pd.read_csv(file)
                list = []
                for item in csv.as_matrix()[::-1]:
                    list.append(float(item[1]) * 0.01)
                self.option_iv.append(np.asarray(list))
            elif file == 'MMR.csv':
                csv = pd.read_csv(file)
                list_0 = []
                list_1 = []
                for item in csv.as_matrix()[5:][::-1]:
                    list_0.append(item[0])
                    list_1.append(float(item[1]) * 0.01)
                self.option_ir_date.append(np.asarray(list_0))
                self.option_ir.append(np.asarray(list_1))
            else:
                csv = pd.read_csv(file)
                list_0 = []
                list_1 = []
                for item in csv.as_matrix()[6:][::-1]:
                    list_0.append(item[0])
                    list_1.append((float(item[1]) + float(item[2])) / 2)
                self.option_date.append(np.asarray(list_0))
                self.option_price.append(np.asarray(list_1))
        self.option_date = np.asarray(self.option_date)
        self.option_price = np.asarray(self.option_price)
        self.option_greeks = np.asarray(self.option_greeks)
        self.option_iv = np.asarray(self.option_iv)
        self.option_ir = np.asarray(self.option_ir)
        self.option_ir_date = np.asarray(self.option_ir_date)


    ## Calculate portfolio value

    # Find the last trading day per week

    def lastday_of_week(self, current_friday, date_list):
        current_friday_datetime = datetime.datetime.strptime(current_friday, "%m/%d/%y").date()
        last_date = None
        n = 0
        while n < 7:
            if (current_friday_datetime + datetime.timedelta(days=7 - n)).strftime('%-m/%-d/%y') in date_list:
                last_date = (current_friday_datetime + datetime.timedelta(days=7 - n)).strftime('%-m/%-d/%y')
                break
            else:
                n += 1
        return last_date

    def cal_weekly_index(self):
        weekly_date = []
        weekly_date.append(self.date_list[0][0])
        weekly_date.append(self.date_list[0][4])
        current_friday = self.date_list[0][4]
        # Get the last trading day per week
        while datetime.datetime.strptime(current_friday, "%m/%d/%y") \
                    < datetime.datetime.strptime('4/19/18', "%m/%d/%y"):
            last_date = self.lastday_of_week(current_friday, self.date_list[0])
            current_friday = (datetime.datetime.strptime(current_friday, "%m/%d/%y").date()
                              + datetime.timedelta(days=7)).strftime('%-m/%-d/%y')
            if last_date == None:
                continue
            else:
                weekly_date.append(last_date)
        weekly_date = weekly_date[0:-1]
        weekly_date.append('4/19/18')
        weekly_date = np.asarray(weekly_date)
        self.weekly_date = weekly_date
        index = []
        for i in range(len(weekly_date)):
            index.append(np.nonzero(self.date_list[0] == weekly_date[i])[0][0])
        index = index[0:-1]
        temp = np.nonzero(self.date_list[0] == '4/19/18')[0][0]
        index.append(temp)
        self.weekly_index = np.asarray(index)

        # lambda list range from 2/28/18 to 4/20/18
        weight = 1/15
        index_ = np.nonzero(self.date_list[0] == '2/28/18')[0][0]
        asset_list = [item[index_] for item in self.price_list]
        self.lambda_.append(self.stock_amount[0] * weight / np.asarray(asset_list))
        for j in self.weekly_index[113:]:
            self.stock_amount.append(sum(np.asarray([i[j] for i in self.price_list])
                                    * np.asarray(self.lambda_[-1])))
            asset_list = [item[j] for item in self.price_list]
            self.lambda_.append(self.stock_amount[-1] * weight / np.asarray(asset_list))
        self.lambda_ = np.asarray(self.lambda_)
        self.stock_amount = np.asarray(self.stock_amount)



    def cal_weekly_bond_index(self):
        for i in range(len(self.bond_date)):
            weekly_date = []
            if i == 0:
                weekly_date.append(self.bond_date[i][2])
                current_friday = self.bond_date[i][2]
            else:
                weekly_date.append(self.bond_date[i][2])
                current_friday = self.bond_date[i][2]
            # Get the last trading day per week
            while datetime.datetime.strptime(current_friday, "%m/%d/%y") \
                    < datetime.datetime.strptime('4/19/18', "%m/%d/%y"):
                last_date = self.lastday_of_week(current_friday, self.bond_date[i])
                current_friday = (datetime.datetime.strptime(current_friday, "%m/%d/%y").date()
                                  + datetime.timedelta(days=7)).strftime('%-m/%-d/%y')
                if last_date == None:
                    continue
                else:
                    weekly_date.append(last_date)
            weekly_date = np.asarray(weekly_date)
            self.bond_weekly_date.append(weekly_date)
            index = []
            for j in range(len(weekly_date)):
                index.append(np.nonzero(self.bond_date[i] == weekly_date[j])[0][0])
            self.bond_weekly_index.append(np.asarray(index))
        self.bond_weekly_date = np.asarray(self.bond_weekly_date)
        self.bond_weekly_index = np.asarray(self.bond_weekly_index)

    def cal_weekly_option_index(self):
        for i in range(len(self.option_date)):
            weekly_date = []
            if i == 0:
                weekly_date.append(self.option_date[i][4])
                current_friday = self.option_date[i][4]
            elif i == 1:
                weekly_date.append(self.option_date[i][0])
                weekly_date.append(self.option_date[i][3])
                current_friday = self.option_date[i][3]
            elif i == 2:
                weekly_date.append(self.option_date[i][4])
                current_friday = self.option_date[i][4]
            # Get the last trading day per week
            while datetime.datetime.strptime(current_friday, "%m/%d/%y") \
                    < datetime.datetime.strptime('4/19/18', "%m/%d/%y"):
                last_date = self.lastday_of_week(current_friday, self.option_date[i])
                current_friday = (datetime.datetime.strptime(current_friday, "%m/%d/%y").date()
                                  + datetime.timedelta(days=7)).strftime('%-m/%-d/%y')
                if last_date == None:
                    continue
                else:
                    weekly_date.append(last_date)
            weekly_date = np.asarray(weekly_date)
            self.option_weekly_date.append(weekly_date)
            index = []
            for j in range(len(weekly_date)):
                index.append(np.nonzero(self.option_date[i] == weekly_date[j])[0][0])
            self.option_weekly_index.append(np.asarray(index))
        self.option_weekly_date = np.asarray(self.option_weekly_date)
        self.option_weekly_index = np.asarray(self.option_weekly_index)

    def cal_option_greeks(self, week_num, option):
        if week_num == 0:
            if option == 0:
                index_s = np.nonzero(self.date_list[0] == '2/28/18')[0][0]
                index_r = np.nonzero(self.option_ir_date[0] == '2/28/18')[0][0]
                index_sigma = np.nonzero(self.option_date[0] == '2/28/18')[0][0]
                T_tau = (datetime.datetime.strptime('5/18/18', "%m/%d/%y") - datetime.datetime.strptime('2/28/18', "%m/%d/%y")).days / 50
                s = self.price_list[7][index_s]
                K = 50
                r = self.option_ir[0][index_r]
                sigma = self.option_iv[0][index_sigma]
            elif option == 1:
                index_s = np.nonzero(self.date_list[0] == '2/28/18')[0][0]
                index_r = np.nonzero(self.option_ir_date[0] == '2/28/18')[0][0]
                index_sigma = np.nonzero(self.option_date[1] == '2/28/18')[0][0]
                T_tau =(datetime.datetime.strptime('5/18/18', "%m/%d/%y") -
                                   datetime.datetime.strptime('2/28/18', "%m/%d/%y")).days / 50
                s = self.price_list[9][index_s]
                K = 315
                r = self.option_ir[0][index_r]
                sigma = self.option_iv[1][index_sigma]
            elif option == 2:
                index_s = np.nonzero(self.date_list[0] == '2/28/18')[0][0]
                index_r = np.nonzero(self.option_ir_date[0] == '2/28/18')[0][0]
                index_sigma = np.nonzero(self.option_date[2] == '2/28/18')[0][0]
                T_tau =(datetime.datetime.strptime('5/18/18', "%m/%d/%y") -
                                   datetime.datetime.strptime('2/28/18', "%m/%d/%y")).days / 50
                s = self.XEO_price[index_s]
                K = 1220
                r = self.option_ir[0][index_r]
                sigma = self.option_iv[2][index_sigma]
        else:
            option_ir_index = []
            T_tau = (datetime.datetime.strptime('5/18/18', "%m/%d/%y") -
                     datetime.datetime.strptime(self.weekly_date[112 + week_num], "%m/%d/%y")).days / 50
            for date in self.option_weekly_date[0]:
                index = np.nonzero(self.option_ir_date[0] == date)[0][0]
                option_ir_index.append(index)
            option_ir_index = np.asarray(option_ir_index)
            if option == 0:
                index_s = np.nonzero(self.date_list[0] == self.weekly_date[112 + week_num])[0][0]
                index_r = option_ir_index[22 + week_num]
                index_sigma = np.nonzero(self.option_date[0] == self.option_weekly_date[0][22 + week_num])[0][0]
                s = self.price_list[7][index_s]
                K = 50
                r = self.option_ir[0][index_r]
                sigma = self.option_iv[0][index_sigma]
            elif option == 1:
                index_s = np.nonzero(self.date_list[0] == self.weekly_date[112 + week_num])[0][0]
                index_r = option_ir_index[22 + week_num]
                index_sigma = np.nonzero(self.option_date[1] == self.option_weekly_date[1][1 + week_num])[0][0]
                s = self.price_list[9][index_s]
                K = 315
                r = self.option_ir[0][index_r]
                sigma = self.option_iv[1][index_sigma]
            elif option == 2:
                index_s = np.nonzero(self.date_list[0] == self.weekly_date[112 + week_num])[0][0]
                index_r = option_ir_index[22 + week_num]
                index_sigma = np.nonzero(self.option_date[2] == self.option_weekly_date[2][4 + week_num])[0][0]
                s = self.XEO_price[index_s]
                K = 1220
                r = self.option_ir[0][index_r]
                sigma = self.option_iv[2][index_sigma]
        # sigma = 0.1407
        # r = 0.0235
        d_1 = (np.log(s/K) + (r + 0.5*(sigma**2))*(T_tau)) / (sigma*np.sqrt(T_tau))
        d_2 = d_1 - sigma*np.sqrt(T_tau)
        phi_d_1 = norm.pdf(d_1)
        phi_d_2 = norm.pdf(d_2)
        # theta = self.option_greeks[option][week_num][0]
        # delta = self.option_greeks[option][week_num][1]
        # rho = self.option_greeks[option][week_num][2]
        # vega = self.option_greeks[option][week_num][3]

        if option == 0 or option == 1:
            theta = (-(s * phi_d_1 * sigma / (2 * np.sqrt(T_tau))) - (r * K * np.exp(-r * T_tau) * norm.cdf(d_2)))/50
            delta = norm.cdf(d_1)
            rho = K * T_tau * np.exp(-r * T_tau) * norm.cdf(d_2)/100
            vega = s * np.sqrt(T_tau) * phi_d_1
            gamma = vega / (s ** 2 * T_tau * sigma)
            vanna = (vega / s) * (1 - d_1 / (sigma * np.sqrt(T_tau)))
            vomma = vega * (d_1 * d_2 / sigma)
        else:
            theta = (-(s * phi_d_1 * sigma / (2 * np.sqrt(T_tau))) + r * K * np.exp(-r * T_tau) * norm.cdf(-d_2))/50
            delta = - norm.cdf(-d_1)
            rho = -K * T_tau * np.exp(-r * T_tau) * norm.cdf(-d_2)/100
            vega = s * np.sqrt(T_tau) * phi_d_1
            gamma = vega / (s ** 2 * T_tau * sigma)
            vanna = (vega / s) * (1 - d_1 / (sigma * np.sqrt(T_tau)))
            vomma = vega * (d_1 * d_2 / sigma)
        return [theta, delta, rho, vega, gamma, vanna, vomma]


    # def cal_monthly_V_t(self):
    #     # Monthly end-date
    #     end_date = ['1/3/12', '1/31/12', '2/29/12', '3/30/12', '4/30/12', '5/31/12', '6/29/12',
    #                 '7/31/12', '8/31/12', '9/28/12', '10/31/12', '11/30/12', '12/31/12',
    #                 '1/31/13', '2/28/13', '3/28/13', '4/30/13', '5/31/13', '6/28/13',
    #                 '7/31/13', '8/30/13', '9/30/13', '10/31/13', '11/29/13', '12/31/13',
    #                 '1/31/14', '2/28/14', '3/31/14', '4/30/14', '5/30/14', '6/30/14',
    #                 '7/31/14', '8/29/14', '9/30/14', '10/31/14', '11/28/14', '12/31/14',
    #                 '1/30/15', '2/27/15', '3/31/15', '4/30/15', '5/29/15', '6/30/15',
    #                 '7/31/15', '8/31/15', '9/30/15', '10/30/15', '11/30/15', '12/31/15',
    #                 '1/29/16', '2/29/16', '3/31/16', '4/29/16', '5/31/16', '6/30/16',
    #                 '7/29/16', '8/31/16', '9/30/16', '10/31/16', '11/30/16', '12/30/16',
    #                 '1/31/17', '2/28/17', '3/31/17', '4/28/17', '5/31/17', '6/30/17',
    #                 '7/31/17', '8/31/17', '9/29/17', '10/31/17', '11/30/17', '12/29/17',
    #                 '1/31/18', '2/28/18', '3/29/18', '4/30/18']
    #     self.monthly_end_date = end_date
    #     # Find index of first 2 years
    #     index = []
    #     for i in range(len(end_date)):
    #         index.append(np.nonzero(self.date_list[0] == end_date[i])[0][0])
    #     self.monthly_index = index
    #     lambda_dict = {}
    #     # Get lambda list
    #     weight = 1 / 15  # N = 15
    #     for i in range(len(end_date)):
    #         asset_list = [item[index[i]] for item in self.price_list]
    #         if i == 0:
    #             lambda_dict[end_date[i]] = self.ic * weight / np.asarray(asset_list)
    #         else:
    #             lambda_dict[end_date[i]] = sum(lambda_dict[end_date[i - 1]] * np.asarray(asset_list)) * weight / np.asarray(asset_list)
    #     self.lambda_dict = lambda_dict
    #     # Get portfolio value list
    #     V_t = [sum(np.asarray([item[0] for item in self.price_list]) * lambda_dict[end_date[0]])]
    #     for i in range(len(index) - 1):
    #         for j in range(index[i] + 1, index[i + 1] + 1):
    #             V_t.append(sum(np.asarray([item[j] for item in self.price_list]) * lambda_dict[end_date[i]]))
    #     self.V_t = np.asarray(V_t)

    def cal_weekly_V_t(self, week_num):
        # Weekly V_t
        lambda_dict = {}
        # Get lambda list
        weight = 1 / 15  # N = 15
        index = self.weekly_index
        end_date = self.weekly_date
        for i in range(week_num, len(end_date)):
            if i == week_num:
                lambda_dict[end_date[i]] = self.lambda_[week_num]
            else:
                asset_list = [item[index[i]] for item in self.price_list]
                lambda_dict[end_date[i]] = sum(lambda_dict[end_date[i - 1]] * np.asarray(asset_list)) * weight / np.asarray(asset_list)
        self.lambda_dict = lambda_dict

        # Get portfolio value list
        # V_t = [sum(np.asarray([item[index[week_num]] for item in self.price_list]) * lambda_dict[end_date[week_num]])]
        # for i in range(week_num, 112 + week_num):
        #     for j in range(index[i] + 1, index[i + 1] + 1):
        #         V_t.append(sum(np.asarray([item[j] for item in self.price_list]) * lambda_dict[end_date[i]]))
        V_t = [sum(np.asarray([item[index[i]] for item in self.price_list]) * lambda_dict[end_date[i]])
               for i in range(week_num, 113 + week_num)]
        self.V_t = np.asarray(V_t)



    ## Calibrate parameters
    # week_num from 0 to 7 !!!
    # stock
    def calibrate_stock_weekly(self, week_num):
        self.cal_weekly_V_t(week_num)
        weight = 1 / 15
        mu = []
        temp = []
        for i in range(len(self.log_return)):
            mu.append(np.mean(self.log_return[i][self.weekly_index[week_num:(113+week_num)]]))
            temp.append(self.log_return[i][self.weekly_index[week_num:(113+week_num)]])
        self.mu = mu
        cov_matrix = np.cov(temp)
        self.cov_matrix = cov_matrix
        ## Fitted by normal
        X_normal = np.random.multivariate_normal(mu, cov_matrix, 10000)
        L = []
        for j in range(len(X_normal.T[0])):
            if week_num==0:
                index = np.nonzero(self.date_list[0] == '2/28/18')[0][0]
                L.append(-sum(
                    self.lambda_[week_num]
                    * np.asarray([item[index] for item in self.price_list])
                    * (np.exp(np.asarray(X_normal[j])) - 1)
                ))
            else:
                L.append(-sum(
                    self.lambda_[week_num]
                    * np.asarray([item[self.weekly_index[112 + week_num]] for item in self.price_list])
                    * (np.exp(np.asarray(X_normal[j])) - 1)
                ))
        L = np.asarray(L)
        if week_num==0:
            index = np.nonzero(self.date_list[0] == '2/28/18')[0][0]
            V_t = sum(np.asarray([item[index] for item in self.price_list]) * self.lambda_[0])
            L_delta = np.array([sum(-weight * V_t * np.asarray(X_normal[item]))
                                for item in range(len(X_normal.T[0]))]
                                )
        else:
            V_t = sum(np.asarray([item[self.weekly_index[112 + week_num]] for item in self.price_list]) * self.lambda_[week_num])
            L_delta = np.array([sum(-weight * V_t * np.asarray(X_normal[item]))
                                for item in range(len(X_normal.T[0]))]
                               )
        L_delta = np.asarray(L_delta)
        ## Fitted by t-student
        L_act = [-(self.V_t[i + 1] - self.V_t[i]) for i in range(len(self.V_t) - 1)]
        self.L_act = np.asarray(L_act)
        parameters = t.fit(np.asarray(L_act))
        L_t = t.rvs(parameters[0], parameters[1], parameters[2], 10000)
        return [np.asarray(L), np.asarray(L_delta), np.asarray(L_t)]

    #bond
    def calibrate_bond_weekly(self, week_num):
        # Linearized normal
        start_index = [np.nonzero(self.bond_date[0] == '2/28/18')[0][0],
                       np.nonzero(self.bond_date[1] == '2/28/18')[0][0]]
        self.bond_lambda = [265000 / self.bond_price[0][start_index[0]], 255000 / self.bond_price[1][start_index[1]]]
        self.bond_lambda = np.asarray(self.bond_lambda)
        difference_0 = [self.bond_yield[0][self.bond_weekly_index[0][i + 1]] - self.bond_yield[0][self.bond_weekly_index[0][i]]
                        for i in range(week_num, 32 + week_num - 1)]
        difference_1 = [
            self.bond_yield[1][self.bond_weekly_index[1][i + 1]] - self.bond_yield[1][self.bond_weekly_index[1][i]]
            for i in range(week_num, 32 + week_num - 1)]
        mu = np.array([np.mean(difference_0), np.mean(difference_1)])
        temp = np.array([difference_0, difference_1])
        cov_matrix = np.cov(temp)
        X_normal = np.random.multivariate_normal(mu, cov_matrix, 10000)
        L = []
        for i in range(len(X_normal.T[0])):
            if week_num==0:
                index = np.nonzero(self.bond_date[0] == '2/28/18')[0][0]
                T_tau = np.array([(datetime.datetime.strptime('8/15/19', "%m/%d/%y") - datetime.datetime.strptime('2/28/18', "%m/%d/%y")).days / (50),
                                  (datetime.datetime.strptime('7/15/22', "%m/%d/%y") - datetime.datetime.strptime(
                                      '2/28/18', "%m/%d/%y")).days / (50)]
                                 )
                L.append(-sum(
                    self.bond_lambda
                    * np.asarray([item[index] for item in self.bond_price])
                    * (np.asarray([item[index] for item in self.bond_yield])/50 - T_tau * X_normal[i])
                ))
            else:
                T_tau = np.array([(datetime.datetime.strptime('8/15/19', "%m/%d/%y") - datetime.datetime.strptime(
                    self.bond_weekly_date[0][31 + week_num], "%m/%d/%y")).days / (50),
                                  (datetime.datetime.strptime('7/15/22', "%m/%d/%y") - datetime.datetime.strptime(
                                      self.bond_weekly_date[0][31 + week_num], "%m/%d/%y")).days / (50)]
                                 )
                L.append(-sum(
                    self.bond_lambda
                    * np.asarray([item[self.bond_weekly_index[0][31 + week_num]] for item in self.bond_price])
                    * (np.asarray([item[self.bond_weekly_index[0][31 + week_num]] for item in self.bond_yield])/50 - T_tau * X_normal[i])
                ))
        L_delta = np.asarray(L)
        # t distribution
        V_t = self.bond_lambda[0]*self.bond_price[0] + self.bond_lambda[1]*self.bond_price[1]
        L_act = [-(V_t[i + 1] - V_t[i]) for i in self.bond_weekly_index[0][week_num:(31 + week_num)]]
        parameters = t.fit(L_act)
        L_t = t.rvs(parameters[0], parameters[1], parameters[2], 10000)
        return [np.asarray(L_delta), np.asarray(L_t)]

    #option
    def calibrate_option_weekly(self, week_num):
        L_delta = []
        L_quadratic = []
        L_t = []
        for i in [0, 2]:
            # Linearized normal
            if i == 0:
                option_ir_index = []
                for date in self.option_weekly_date[i]:
                    index = np.nonzero(self.option_ir_date[0] == date)[0][0]
                    option_ir_index.append(index)
                option_ir_index = np.asarray(option_ir_index)
                mu = np.array([np.mean(self.log_return[7][self.weekly_index[week_num:(113 + week_num)]]),
                               np.mean(np.asarray([self.option_ir[0][option_ir_index[item + 1]]-self.option_ir[0][option_ir_index[item]]
                                       for item in range(week_num, 23 + week_num - 1)])),
                               np.mean(
                                   np.asarray([self.option_iv[i][self.option_weekly_index[i][item + 1]] - self.option_iv[i][self.option_weekly_index[i][item]]
                                   for item in range(week_num, 23 + week_num - 1)]))
                               ])
                std = np.array([np.std(self.log_return[7][self.weekly_index[week_num:(113 + week_num)]]),
                               np.std(np.asarray([self.option_ir[0][option_ir_index[item + 1]]-self.option_ir[0][option_ir_index[item]]
                                       for item in range(week_num, 23 + week_num - 1)])),
                               np.std(
                                   np.asarray([self.option_iv[i][self.option_weekly_index[i][item + 1]] - self.option_iv[i][self.option_weekly_index[i][item]]
                                   for item in range(week_num, 23 + week_num - 1)]))
                               ])
            elif i == 1:
                option_ir_index = []
                for date in self.option_weekly_date[i]:
                    index = np.nonzero(self.option_ir_date[0] == date)[0][0]
                    option_ir_index.append(index)
                option_ir_index = np.asarray(option_ir_index)
                mu = np.array([np.mean(self.log_return[9][self.weekly_index[week_num:(113 + week_num)]]),
                               np.mean(np.asarray([self.option_ir[0][option_ir_index[item + 1]] - self.option_ir[0][
                                   option_ir_index[item]]
                                                   for item in range(week_num, 2 + week_num - 1)])),
                               np.mean(
                                   np.asarray([self.option_iv[i][self.option_weekly_index[i][item + 1]] -
                                               self.option_iv[i][self.option_weekly_index[i][item]]
                                               for item in range(week_num, 2 + week_num - 1)]))
                               ])
                std = np.array([np.std(self.log_return[9][self.weekly_index[week_num:(113 + week_num)]]),
                                np.std(np.asarray([self.option_ir[0][option_ir_index[item + 1]] - self.option_ir[0][
                                    option_ir_index[item]]
                                                   for item in range(week_num, 2 + week_num - 1)])),
                                np.std(
                                    np.asarray([self.option_iv[i][self.option_weekly_index[i][item + 1]] -
                                                self.option_iv[i][self.option_weekly_index[i][item]]
                                                for item in range(week_num, 2 + week_num - 1)]))
                                ])
            elif i == 2:
                option_ir_index = []
                for date in self.option_weekly_date[i]:
                    index = np.nonzero(self.option_ir_date[0] == date)[0][0]
                    option_ir_index.append(index)
                option_ir_index = np.asarray(option_ir_index)
                mu = np.array([np.mean(self.XEO_log_return[self.weekly_index[week_num:(113 + week_num)]]),
                               np.mean(np.asarray([self.option_ir[0][option_ir_index[item + 1]] - self.option_ir[0][
                                   option_ir_index[item]]
                                                   for item in range(week_num, 5 + week_num - 1)])),
                               np.mean(
                                   np.asarray([self.option_iv[i][self.option_weekly_index[i][item + 1]] -
                                               self.option_iv[i][self.option_weekly_index[i][item]]
                                               for item in range(week_num, 5 + week_num - 1)]))
                               ])
                std = np.array([np.std(self.XEO_log_return[self.weekly_index[week_num:(113 + week_num)]]),
                                np.std(np.asarray([self.option_ir[0][option_ir_index[item + 1]] - self.option_ir[0][
                                    option_ir_index[item]]
                                                   for item in range(week_num, 5 + week_num - 1)])),
                                np.std(
                                    np.asarray([self.option_iv[i][self.option_weekly_index[i][item + 1]] -
                                                self.option_iv[i][self.option_weekly_index[i][item]]
                                                for item in range(week_num, 5 + week_num - 1)]))
                                ])
            X_normal = np.array([np.random.normal(mu[0], std[0], 10000), np.random.normal(mu[1], std[1], 10000),
                                 np.random.normal(mu[2], std[2], 10000)])
            L_first = []
            L_second = []
            greeks = self.cal_option_greeks(week_num, i)
            for j in range(len(X_normal[0])):
                if week_num == 0:
                    index = np.nonzero(self.date_list[0] == '2/28/18')[0][0]
                    if i == 0:
                        # L_first.append(-(self.option_greeks[i][week_num][0]
                        #            + self.option_greeks[i][week_num][1] * X_normal[0][j] * self.price_list[7][index]
                        #            + self.option_greeks[i][week_num][2] * X_normal[1][j]
                        #            + self.option_greeks[i][week_num][3] * X_normal[2][j]))
                        L_first.append(-(greeks[0]/50
                                         + greeks[1] * X_normal[0][j] * self.price_list[7][index]
                                         + greeks[2] * X_normal[1][j]
                                         + greeks[3] * X_normal[2][j]))
                        L_second.append(
                            -(
                                (greeks[0]/50
                                 + greeks[1] * X_normal[0][j] * self.price_list[7][index]
                                 + greeks[2] * X_normal[1][j]
                                 + greeks[3] * X_normal[2][j])
                                +
                                0.5 * (
                                greeks[4] * (X_normal[0][j]**2) * (self.price_list[7][index]**2) +
                                2 * greeks[5] * self.price_list[7][index] * X_normal[0][j] * X_normal[2][j] +
                                greeks[6] * (X_normal[2][j]**2)
                                )
                            )
                        )
                    elif i == 1:
                        # L_first.append(-(self.option_greeks[i][week_num][0]
                        #                  + self.option_greeks[i][week_num][1] * X_normal[0][j] * self.price_list[9][
                        #                      index]
                        #                  + self.option_greeks[i][week_num][2] * X_normal[1][j]
                        #                  + self.option_greeks[i][week_num][3] * X_normal[2][j]))
                        L_first.append(-(greeks[0]/50
                                   + greeks[1] * X_normal[0][j] * self.price_list[9][index]
                                   + greeks[2] * X_normal[1][j]
                                   + greeks[3] * X_normal[2][j]))
                        L_second.append(
                                -(
                                        (greeks[0]/50
                                         + greeks[1] * X_normal[0][j] * self.price_list[9][index]
                                         + greeks[2] * X_normal[1][j]
                                         + greeks[3] * X_normal[2][j])
                                        +
                                        0.5 * (
                                                greeks[4] * (X_normal[0][j] ** 2) * (self.price_list[9][index] ** 2) +
                                                2 * greeks[5] * self.price_list[9][index] * X_normal[0][j] *
                                                X_normal[2][j] +
                                                greeks[6] * (X_normal[2][j] ** 2)
                                        )
                                )
                        )
                    elif i == 2:
                        # L_first.append(-(self.option_greeks[i][week_num][0]
                        #                  + self.option_greeks[i][week_num][1] * X_normal[0][j] * self.XEO_price[index]
                        #                  + self.option_greeks[i][week_num][2] * X_normal[1][j]
                        #                  + self.option_greeks[i][week_num][3] * X_normal[2][j]))
                        L_first.append(-(greeks[0]/50
                                   + greeks[1] * X_normal[0][j] * self.XEO_price[index]
                                   + greeks[2] * X_normal[1][j]
                                   + greeks[3] * X_normal[2][j]))
                        L_second.append(
                                -(
                                        (greeks[0]/50
                                         + greeks[1] * X_normal[0][j] * self.XEO_price[index]
                                         + greeks[2] * X_normal[1][j]
                                         + greeks[3] * X_normal[2][j])
                                        +
                                        0.5 * (
                                                greeks[4] * (X_normal[0][j] ** 2) * (self.XEO_price[index] ** 2) +
                                                2 * greeks[5] * self.XEO_price[index] * X_normal[0][j] *
                                                X_normal[2][j] +
                                                greeks[6] * (X_normal[2][j] ** 2)
                                        )
                                )
                        )
                else:
                    index = self.weekly_index[112 + week_num]
                    if i == 0:
                        # L_first.append(-(self.option_greeks[i][week_num][0]
                        #                  + self.option_greeks[i][week_num][1] * X_normal[0][j] * self.price_list[7][
                        #                      index]
                        #                  + self.option_greeks[i][week_num][2] * X_normal[1][j]
                        #                  + self.option_greeks[i][week_num][3] * X_normal[2][j]))
                        L_first.append(-(greeks[0]/50
                                   + greeks[1] * X_normal[0][j] * self.price_list[7][index]
                                   + greeks[2] * X_normal[1][j]
                                   + greeks[3] * X_normal[2][j]))
                        L_second.append(
                            -(
                                (greeks[0]/50
                                 + greeks[1] * X_normal[0][j] * self.price_list[7][index]
                                 + greeks[2] * X_normal[1][j]
                                 + greeks[3] * X_normal[2][j])
                                +
                                0.5 * (
                                greeks[4] * (X_normal[0][j]**2) * (self.price_list[7][index]**2) +
                                2 * greeks[5] * self.price_list[7][index] * X_normal[0][j] * X_normal[2][j] +
                                greeks[6] * (X_normal[2][j]**2)
                                )
                            )
                        )
                    elif i == 1:
                        # L_first.append(-(self.option_greeks[i][week_num][0]
                        #                  + self.option_greeks[i][week_num][1] * X_normal[0][j] * self.price_list[9][
                        #                      index]
                        #                  + self.option_greeks[i][week_num][2] * X_normal[1][j]
                        #                  + self.option_greeks[i][week_num][3] * X_normal[2][j]))
                        L_first.append(-(greeks[0]/50
                                   + greeks[1] * X_normal[0][j] * self.price_list[9][index]
                                   + greeks[2] * X_normal[1][j]
                                   + greeks[3] * X_normal[2][j]))
                        L_second.append(
                                -(
                                        (greeks[0]/50
                                         + greeks[1] * X_normal[0][j] * self.price_list[9][index]
                                         + greeks[2] * X_normal[1][j]
                                         + greeks[3] * X_normal[2][j])
                                        +
                                        0.5 * (
                                                greeks[4] * (X_normal[0][j] ** 2) * (self.price_list[9][index] ** 2) +
                                                2 * greeks[5] * self.price_list[9][index] * X_normal[0][j] *
                                                X_normal[2][j] +
                                                greeks[6] * (X_normal[2][j] ** 2)
                                        )
                                )
                        )
                    elif i == 2:
                        # L_first.append(-(self.option_greeks[i][week_num][0]
                        #                  + self.option_greeks[i][week_num][1] * X_normal[0][j] * self.XEO_price[index]
                        #                  + self.option_greeks[i][week_num][2] * X_normal[1][j]
                        #                  + self.option_greeks[i][week_num][3] * X_normal[2][j]))
                        L_first.append(-(greeks[0]/50
                                   + greeks[1] * X_normal[0][j] * self.XEO_price[index]
                                   + greeks[2] * X_normal[1][j]
                                   + greeks[3] * X_normal[2][j]))
                        L_second.append(
                                -(
                                        (greeks[0]/50
                                         + greeks[1] * X_normal[0][j] * self.XEO_price[index]
                                         + greeks[2] * X_normal[1][j]
                                         + greeks[3] * X_normal[2][j])
                                        +
                                        0.5 * (
                                                greeks[4] * (X_normal[0][j] ** 2) * (self.XEO_price[index] ** 2) +
                                                2 * greeks[5] * self.XEO_price[index] * X_normal[0][j] *
                                                X_normal[2][j] +
                                                greeks[6] * (X_normal[2][j] ** 2)
                                        )
                                )
                        )
            L_delta.append(np.asarray(L_first))
            L_quadratic.append(np.asarray(L_second))
        #L_delta = -6 * L_delta[0] + (-1) * L_delta[1] + L_delta[2]
        L_delta = -600 * np.asarray(L_delta[0]) + 100 * np.asarray(L_delta[1])
        #L_quadratic = -6 * L_quadratic[0] + (-1) * L_quadratic[1] + L_quadratic[2]
        L_quadratic = -600 * np.asarray(L_quadratic[0]) + 100 * np.asarray(L_quadratic[1])

        # t distribution
        index_0 = np.nonzero(self.option_date[0] == '1/26/18')[0][0]
        option_price_0 = self.option_price[0][index_0:]
        #index_1 = np.nonzero(self.option_weekly_date[1] == '1/26/18')[0][0]
        #option_price_1 = self.option_price[1]
        index_2 = np.nonzero(self.option_date[2] == '1/26/18')[0][0]
        option_price_2 = self.option_price[2][index_2:]
        V_t = - 600 * np.asarray(option_price_0) + 100 * np.asarray(option_price_2)
        L_act = [-(V_t[i + 1] - V_t[i]) for i in self.option_weekly_index[0][0:(4 + week_num)]]
        parameters = t.fit(L_act)
        L_t = t.rvs(parameters[0], parameters[1], parameters[2], 10000)
        return [np.asarray(L_delta), np.asarray(L_quadratic), np.asarray(L_t)]

    def analyze_all_8_weeks(self):
        # Actual Loss
        real_loss_8_rebalanced = np.array([self.stock_amount[i + 1] - self.stock_amount[i] for i in range(len(self.stock_amount) - 1)])
        rela_V_t = np.array([sum(np.asarray([item[542] for item in self.price_list])
                                                 * np.asarray(self.lambda_[0]))] +
                                             [sum(np.asarray([item[i] for item in self.price_list])
                                                  * np.asarray(self.lambda_[0])) for i in self.weekly_index[113:]])
        real_loss_8_no_rebalanced = np.array([rela_V_t[i + 1] - rela_V_t[i] for i in range(len(rela_V_t) - 1)])
        # Linearized model
        L_delta_VaR_95 = []
        L_delta_CVaR_95 = []
        L_delta_VaR_99 = []
        L_delta_CVaR_99 = []
        # Linearized mix Quadratic
        L_mix_VaR_95 = []
        L_mix_CVaR_95 = []
        L_mix_VaR_99 = []
        L_mix_CVaR_99 = []
        # T-student
        L_t_VaR_95 = []
        L_t_CVaR_95 = []
        L_t_VaR_99 = []
        L_t_CVaR_99 = []
        for i in range(8):
            L_0, L_delta_0, L_t_0 = self.calibrate_stock_weekly(i)
            L_delta_1, L_t_1 = self.calibrate_bond_weekly(i)
            L_delta_2, L_quadratic, L_t_2 = self.calibrate_option_weekly(i)
            L_delta = L_delta_0 + L_delta_1 + L_delta_2
            L_mix = L_delta_0 + L_delta_1 + L_quadratic
            L_t = L_t_0 + L_t_1 + L_t_2
            ## Linearized
            # Cal VaR and CVaR
            mu, sigma = norm.fit(L_delta)
            L_delta_VaR_95.append(norm.ppf(0.95, mu, sigma))
            L_delta_CVaR_95.append(mu + sigma * (norm.pdf(norm.ppf(0.95)) / (1 - 0.95)))
            L_delta_VaR_99.append(norm.ppf(0.99, mu, sigma))
            L_delta_CVaR_99.append(mu + sigma * (norm.pdf(norm.ppf(0.99)) / (1 - 0.99)))
            ## Linearized mix Quadratic
            # Cal VaR and CVaR
            mu, sigma = norm.fit(L_mix)
            L_mix_VaR_95.append(norm.ppf(0.95, mu, sigma))
            L_mix_CVaR_95.append(mu + sigma * (norm.pdf(norm.ppf(0.95)) / (1 - 0.95)))
            L_mix_VaR_99.append(norm.ppf(0.99, mu, sigma))
            L_mix_CVaR_99.append(mu + sigma * (norm.pdf(norm.ppf(0.99)) / (1 - 0.99)))
            ## T student
            # Cal VaR and CVaR
            L_t_fit_df, L_t_fit_mu, L_t_fit_sigma = t.fit(L_t)
            L_t_VaR_95.append(t.ppf(0.95, L_t_fit_df, L_t_fit_mu, L_t_fit_sigma))
            L_t_CVaR_95.append(L_t_fit_mu + L_t_fit_sigma * ((t.pdf(t.ppf(0.95, L_t_fit_df), L_t_fit_df) / (1 - 0.95))) * (
            ((L_t_fit_df + (t.ppf(0.95, L_t_fit_df)) ** 2) / (L_t_fit_df - 1))))
            L_t_VaR_99.append(t.ppf(0.99, L_t_fit_df, L_t_fit_mu, L_t_fit_sigma))
            L_t_CVaR_99.append(L_t_fit_mu + L_t_fit_sigma * ((t.pdf(t.ppf(0.99, L_t_fit_df), L_t_fit_df) / (1 - 0.99))) * (
            ((L_t_fit_df + (t.ppf(0.99, L_t_fit_df)) ** 2) / (L_t_fit_df - 1))))
        # # Save lists
        # with open("final_project_VaR_CVaR.txt", "wb") as fp:  # Pickling
        #     pickle.dump([[L_delta_VaR_95, L_mix_VaR_95, L_t_VaR_95, L_t_VaR_95, Yaxis_VaR_regression_95],
        #                  [G_normal_CVaR_95, G_student_CVaR_95, L_CVaR_95, L_t_CVaR_95, Yaxis_ES_regression_95],
        #                  [G_normal_VaR_99, G_student_VaR_99, L_VaR_99, L_t_VaR_99, Yaxis_VaR_regression_99],
        #                  [G_normal_CVaR_99, G_student_CVaR_99, L_CVaR_99, L_t_CVaR_99, Yaxis_ES_regression_99]], fp)
        # Plot
        xaxis = range(len(L_delta_VaR_95))
        # VaR 0.95 case
        fig, ax = plt.subplots(1,1)
        plt.plot(xaxis, L_delta_VaR_95, label='Linearized Loss fitted by normal')
        plt.plot(xaxis, L_mix_VaR_95, label='Mixed Loss fitted by normal')
        #plt.plot(xaxis, L_t_VaR_95, label='Loss fitted by t student')
        plt.scatter(xaxis, real_loss_8_rebalanced, label='Actual Loss')
        plt.legend()
        plt.xlabel("Weeks")
        plt.ylabel("Values")
        plt.title("VaR_0.95 among 2 models versus time")
        fig.canvas.draw()
        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels[0] = '2/28/18'
        labels[1] = '3/2/18'
        labels[2] = '3/9/18'
        labels[3] = '3/16/18'
        labels[4] = '3/23/18'
        labels[5] = '3/29/18'
        labels[6] = '4/6/18'
        labels[7] = '4/13/18'
        labels[8] = '4/19/18'
        ax.set_xticklabels(labels, rotation=45)
        plt.show()

        # VaR 0.99 case
        fig, ax = plt.subplots(1, 1)
        plt.plot(xaxis, L_delta_VaR_99, label='Linearized Loss fitted by normal')
        plt.plot(xaxis, L_mix_VaR_99, label='Mixed Loss fitted by normal')
        #plt.plot(xaxis, L_t_VaR_99, label='Loss fitted by t student')
        plt.scatter(xaxis, real_loss_8_rebalanced, label='Actual Loss')
        plt.legend()
        plt.xlabel("Weeks")
        plt.ylabel("Values")
        plt.title("VaR_0.99 among 2 models versus time")
        fig.canvas.draw()
        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels[0] = '2/28/18'
        labels[1] = '3/2/18'
        labels[2] = '3/9/18'
        labels[3] = '3/16/18'
        labels[4] = '3/23/18'
        labels[5] = '3/29/18'
        labels[6] = '4/6/18'
        labels[7] = '4/13/18'
        labels[8] = '4/19/18'
        ax.set_xticklabels(labels, rotation=45)
        plt.show()

        # CVaR 0.95 case
        fig, ax = plt.subplots(1, 1)
        plt.plot(xaxis, L_delta_CVaR_95, label='Linearized Loss fitted by normal')
        plt.plot(xaxis, L_mix_CVaR_95, label='Mixed Loss fitted by normal')
        # plt.plot(xaxis, L_t_VaR_99, label='Loss fitted by t student')
        plt.scatter(xaxis, real_loss_8_rebalanced, label='Actual Loss')
        plt.legend()
        plt.xlabel("Weeks")
        plt.ylabel("Values")
        plt.title("CVaR_0.95 among 2 models versus time")
        fig.canvas.draw()
        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels[0] = '2/28/18'
        labels[1] = '3/2/18'
        labels[2] = '3/9/18'
        labels[3] = '3/16/18'
        labels[4] = '3/23/18'
        labels[5] = '3/29/18'
        labels[6] = '4/6/18'
        labels[7] = '4/13/18'
        labels[8] = '4/19/18'
        ax.set_xticklabels(labels, rotation=45)
        plt.show()

        # CVaR 0.99 case
        fig, ax = plt.subplots(1, 1)
        plt.plot(xaxis, L_delta_CVaR_99, label='Linearized Loss fitted by normal')
        plt.plot(xaxis, L_mix_CVaR_99, label='Mixed Loss fitted by normal')
        # plt.plot(xaxis, L_t_VaR_99, label='Loss fitted by t student')
        plt.scatter(xaxis, real_loss_8_rebalanced, label='Actual Loss')
        plt.legend()
        plt.xlabel("Weeks")
        plt.ylabel("Values")
        plt.title("CVaR_0.99 among 2 models versus time")
        fig.canvas.draw()
        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels[0] = '2/28/18'
        labels[1] = '3/2/18'
        labels[2] = '3/9/18'
        labels[3] = '3/16/18'
        labels[4] = '3/23/18'
        labels[5] = '3/29/18'
        labels[6] = '4/6/18'
        labels[7] = '4/13/18'
        labels[8] = '4/19/18'
        ax.set_xticklabels(labels, rotation=45)
        plt.show()

        ## Table
        x = PrettyTable(
            ["VaR_0.95", "3/2/18", '3/9/18', '3/16/18', '3/23/18', '3/29/18', '4/6/18', '4/13/18', '4/19/18'])
        x.align["VaR_0.95"] = "l"
        x.padding_width = 1
        x.add_row(['Linearized', L_delta_VaR_95[0], L_delta_VaR_95[1], L_delta_VaR_95[2], L_delta_VaR_95[3],
                   L_delta_VaR_95[4], L_delta_VaR_95[5], L_delta_VaR_95[6], L_delta_VaR_95[7]])
        x.add_row(['Mixed', L_mix_VaR_95[0], L_mix_VaR_95[1], L_mix_VaR_95[2], L_mix_VaR_95[3],
                   L_mix_VaR_95[4], L_mix_VaR_95[5], L_mix_VaR_95[6], L_mix_VaR_95[7]])
        x.add_row(['T student', L_t_VaR_95[0], L_t_VaR_95[1], L_t_VaR_95[2], L_t_VaR_95[3],
                   L_t_VaR_95[4], L_t_VaR_95[5], L_t_VaR_95[6], L_t_VaR_95[7]])

        y = PrettyTable(
            ["VaR_0.95(%)", "3/2/18", '3/9/18', '3/16/18', '3/23/18', '3/29/18', '4/6/18', '4/13/18', '4/19/18'])
        y.align["VaR_0.95"] = "l"
        y.padding_width = 1
        y.add_row(['Linearized', "%.2f" % (L_delta_VaR_95[0]/1000000*100), "%.2f" % (L_delta_VaR_95[1]/1000000*100), "%.2f" %(L_delta_VaR_95[2]/1000000*100), "%.2f" %(L_delta_VaR_95[3]/1000000*100),
                   "%.2f" %(L_delta_VaR_95[4]/1000000*100), "%.2f" %(L_delta_VaR_95[5]/1000000*100), "%.2f" %(L_delta_VaR_95[6]/1000000*100), "%.2f" %(L_delta_VaR_95[7]/1000000*100)])
        y.add_row(['Mixed', "%.2f" % (L_mix_VaR_95[0]/1000000*100), "%.2f" % (L_mix_VaR_95[1]/1000000*100), "%.2f" %(L_mix_VaR_95[2]/1000000*100), "%.2f" %(L_mix_VaR_95[3]/1000000*100),
                   "%.2f" %(L_mix_VaR_95[4]/1000000*100), "%.2f" %(L_mix_VaR_95[5]/1000000*100), "%.2f" %(L_mix_VaR_95[6]/1000000*100), "%.2f" %(L_mix_VaR_95[7]/1000000*100)])
        y.add_row(['T student', "%.2f" % (L_t_VaR_95[0]/1000000*100), (L_t_VaR_95[1]/1000000*100), (L_t_VaR_95[2]/1000000*100), "%.2f" %(L_t_VaR_95[3]/1000000*100),
                   "%.2f" %(L_t_VaR_95[4]/1000000*100), "%.2f" %(L_t_VaR_95[5]/1000000*100), (L_t_VaR_95[6]/1000000*100), "%.2f" %(L_t_VaR_95[7]/1000000*100)])
        print(x)
        print(y)

    def GARCH_test(self):
        real_loss_8_rebalanced = np.array(
            [self.stock_amount[i + 1] - self.stock_amount[i] for i in range(len(self.stock_amount) - 1)])
        L_act = []
        # GARCH normal
        G_normal_VaR_95 = []
        G_normal_CVaR_95 = []
        G_normal_VaR_99 = []
        G_normal_CVaR_99 = []
        # GARCH student
        G_student_VaR_95 = []
        G_student_CVaR_95 = []
        G_student_VaR_99 = []
        G_student_CVaR_99 = []
        for i in range(8):
            # if i==53 and i==56:
            #    continue
            print("This is " + str(i))
            params = self.get_GARCH_mu_sigma_x(i)
            mu_tplus1_normal = params[0][-1]
            sigma_tplus1_normal = np.sqrt(params[1][-1])
            mu_tplus1_student = params[3][-1]
            sigma_tplus1_student = np.sqrt(params[4][-1])
            df = self.get_ARMA_GARCH_parameters(i, "StudentsT")[3][3]

            G_normal_VaR_95.append(mu_tplus1_normal + sigma_tplus1_normal * norm.ppf(0.95))
            G_normal_CVaR_95.append(mu_tplus1_normal + sigma_tplus1_normal * (norm.pdf(norm.ppf(0.95)) / (1 - 0.95)))
            G_normal_VaR_99.append(mu_tplus1_normal + sigma_tplus1_normal * norm.ppf(0.99))
            G_normal_CVaR_99.append(mu_tplus1_normal + sigma_tplus1_normal * (norm.pdf(norm.ppf(0.99)) / (1 - 0.99)))

            G_student_VaR_95.append(mu_tplus1_student + sigma_tplus1_student * t.ppf(0.95, df))
            G_student_CVaR_95.append(mu_tplus1_student + sigma_tplus1_student
                                     * ((t.pdf(t.ppf(0.95, df), df) / (1 - 0.95))) * (
                                     ((df + (t.ppf(0.95, df)) ** 2) / (df - 1))))
            G_student_VaR_99.append(mu_tplus1_student + sigma_tplus1_student * t.ppf(0.99, df))
            G_student_CVaR_99.append(mu_tplus1_student + sigma_tplus1_student
                                     * ((t.pdf(t.ppf(0.99, df), df) / (1 - 0.99))) * (
                                     ((df + (t.ppf(0.99, df)) ** 2) / (df - 1))))
        xaxis = range(len(G_normal_VaR_95))
        # VaR 0.95 case
        fig, ax = plt.subplots(1, 1)
        plt.plot(xaxis, G_normal_CVaR_95, label='ARMA-GARCH(1,1) normal')
        plt.plot(xaxis, G_student_CVaR_95, label='ARMA-GARCH(1,1) t student')
        # plt.plot(xaxis, L_t_VaR_95, label='Loss fitted by t student')
        plt.scatter(xaxis, real_loss_8_rebalanced, label='Actual Loss')
        plt.legend()
        plt.xlabel("Weeks")
        plt.ylabel("Values")
        plt.title("VaR_0.95 among 2 models versus time")
        fig.canvas.draw()
        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels[0] = '2/28/18'
        labels[1] = '3/2/18'
        labels[2] = '3/9/18'
        labels[3] = '3/16/18'
        labels[4] = '3/23/18'
        labels[5] = '3/29/18'
        labels[6] = '4/6/18'
        labels[7] = '4/13/18'
        labels[8] = '4/19/18'
        ax.set_xticklabels(labels, rotation=45)
        plt.show()

    ## Plot weekly loss distributions

    def plot_dist(self):
        # L, L_delta, L_t = self.calibrate_option_weekly(0)
        # ## Fitted by normal
        # # Loss distribution
        # # PDF
        # plt.subplot(211)
        # plt.title("L fitted by normal", y=1.08)
        # sns.distplot(L, label='PDF')
        # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        # plt.legend()
        # plt.grid(True)
        # # CDF
        # plt.subplot(212)
        # p = np.arange(len(L)) / (len(L) - 1)
        # plt.plot(sorted(L), p, label="CDF")
        # plt.xlabel("loss(unit = $1)")
        # plt.legend()
        # plt.grid(True)
        # plt.show()
        # # Linearized Loss distribution
        # # PDF
        # plt.subplot(211)
        # plt.title("Linearized L fitted by normal", y=1.08)
        # sns.distplot(L_delta, label='PDF')
        # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        # plt.legend()
        # plt.grid(True)
        # # CDF
        # p = np.arange(len(L_delta)) / (len(L_delta) - 1)
        # plt.subplot(212)
        # plt.plot(sorted(L_delta), p, label="CDF")
        # plt.xlabel("loss(unit = $1)")
        # plt.legend()
        # plt.grid(True)
        # plt.show()
        #
        # ## Fitted by t-distribution
        # # PDF
        # plt.subplot(211)
        # plt.title("L fitted by student-t", y=1.08)
        # sns.distplot(L_t, label='PDF')
        # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        # plt.legend()
        # plt.grid(True)
        # # CDF
        # p = np.arange(len(L_t)) / (len(L_t) - 1)
        # plt.subplot(212)
        # plt.plot(sorted(L_t), p, label="CDF")
        # plt.xlabel("loss(unit = $1)")
        # plt.legend()
        # plt.grid(True)
        # plt.show()
        #
        # ## QQ Plot
        # # for L
        # plt.subplot(131)
        # mu, std = norm.fit(L)
        # orderValues = np.sort(L)
        # theoreticalQuantile = []
        # for i in range(len(L)):
        #     theoreticalQuantile.append(norm.ppf(i / (1 + len(L)), mu, std))
        #
        # plt.scatter(orderValues, theoreticalQuantile)
        # plt.plot(theoreticalQuantile, theoreticalQuantile, color='r')
        # plt.title("Loss")
        #
        # # for L_delta
        # plt.subplot(132)
        # mu, std = norm.fit(L_delta)
        # orderValues = np.sort(L_delta)
        # theoreticalQuantile = []
        # for i in range(len(L_delta)):
        #     theoreticalQuantile.append(norm.ppf(i / (1 + len(L_delta)), mu, std))
        #
        # plt.scatter(orderValues, theoreticalQuantile)
        # plt.plot(theoreticalQuantile, theoreticalQuantile, color='r')
        # plt.title("Linearized loss")
        #
        # # for L_t
        # plt.subplot(133)
        # mu, std = norm.fit(L_t)
        # orderValues = np.sort(L_t)
        # theoreticalQuantile = []
        # for i in range(len(L_t)):
        #     theoreticalQuantile.append(norm.ppf(i / (1 + len(L_t)), mu, std))
        #
        # plt.scatter(orderValues, theoreticalQuantile)
        # plt.plot(theoreticalQuantile, theoreticalQuantile, color='r')
        # plt.title("Student-t")
        # plt.show()

        #L, L_delta, L_t = self.calibrate_stock_weekly(4)
        L_0, L_delta_0, L_t_0 = self.calibrate_stock_weekly(0)
        L_delta_1, L_t_1 = self.calibrate_bond_weekly(0)
        L_delta_2, L_quadratic, L_t_2 = self.calibrate_option_weekly(0)
        L_delta = L_delta_0 + L_delta_1 + L_delta_2
        L_mix = L_delta_0 + L_delta_1 + L_quadratic
        L_t_temp = L_t_0 + L_t_1 + L_t_2
        L_t = []
        for i in range(len(L_t_temp)):
            if abs(L_t_temp[i]) <= 300000:
                L_t.append(L_t_temp[i])
        L_t = np.asarray(L_t)
        # ## Fitted by normal
        # # Loss distribution
        # # PDF
        # plt.subplot(211)
        # plt.title("L fitted by normal", y=1.08)
        # plt.hist(L, normed=True)
        # kde = stats.gaussian_kde(L)
        # xaxis = np.linspace(L.min(), L.max(), 1000)
        # plt.plot(xaxis, kde(xaxis), label="PDF")
        # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        # plt.legend()
        # plt.grid(True)
        # # CDF
        # plt.subplot(212)
        # p = np.arange(len(L)) / (len(L) - 1)
        # plt.plot(sorted(L), p, label="CDF")
        # plt.xlabel("loss(unit = $1)")
        # plt.legend()
        # plt.grid(True)
        # plt.show()
        # Linearized Loss distribution
        # PDF
        plt.subplot(211)
        plt.title("Linearized L fitted by normal", y=1.08)
        plt.hist(L_delta, normed=True)
        kde = stats.gaussian_kde(L_delta)
        xaxis_delta = np.linspace(L_delta.min(), L_delta.max(), 1000)
        plt.plot(xaxis_delta, kde(xaxis_delta), label="PDF")
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.legend()
        plt.grid(True)
        # CDF
        p = np.arange(len(L_delta)) / (len(L_delta) - 1)
        plt.subplot(212)
        plt.plot(sorted(L_delta), p, label="CDF")
        plt.xlabel("loss(unit = $1)")
        plt.legend()
        plt.grid(True)
        plt.show()

        ## Fitted by t-distribution
        # PDF
        plt.subplot(211)
        plt.title("L fitted by student-t", y=1.08)
        plt.hist(L_t, normed=True)
        kde = stats.gaussian_kde(L_t)
        xaxis_t = np.linspace(L_t.min(), L_t.max(), 1000)
        plt.plot(xaxis_t, kde(xaxis_t), label="PDF")
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.legend()
        plt.grid(True)
        # CDF
        p = np.arange(len(L_t)) / (len(L_t) - 1)
        plt.subplot(212)
        plt.plot(sorted(L_t), p, label="CDF")
        plt.xlabel("loss(unit = $1)")
        plt.legend()
        plt.grid(True)
        plt.show()

        ## QQ Plot

        # for L_delta
        plt.subplot(131)
        mu, std = norm.fit(L_delta)
        orderValues = np.sort(L_delta)
        theoreticalQuantile = []
        for i in range(len(L_delta)):
            theoreticalQuantile.append(norm.ppf(i / (1 + len(L_delta)), mu, std))

        plt.scatter(orderValues, theoreticalQuantile)
        plt.plot(theoreticalQuantile, theoreticalQuantile, color='r')
        plt.title("Linearized loss")

        # for L mix
        plt.subplot(132)
        mu, std = norm.fit(L_mix)
        orderValues = np.sort(L_mix)
        theoreticalQuantile = []
        for i in range(len(L_mix)):
            theoreticalQuantile.append(norm.ppf(i / (1 + len(L_mix)), mu, std))

        plt.scatter(orderValues, theoreticalQuantile)
        plt.plot(theoreticalQuantile, theoreticalQuantile, color='r')
        plt.title("Loss quadratic")

        # for L_t
        plt.subplot(133)
        mu, std = norm.fit(L_t)
        orderValues = np.sort(L_t)
        theoreticalQuantile = []
        for i in range(len(L_t)):
            theoreticalQuantile.append(norm.ppf(i / (1 + len(L_t)), mu, std))

        plt.scatter(orderValues, theoreticalQuantile)
        plt.plot(theoreticalQuantile, theoreticalQuantile, color='r')
        plt.title("Student-t")
        plt.show()
        a


    ## Compare among normal L, normal L_delta and t-student L
    def compare_normal_t(self):
        for i in range(24, 72):
            L, L_delta, L_t = self.calibrate(i)
            plt.figure(100 + i)
            # PDF
            plt.subplot(211)
            kde = stats.gaussian_kde(L)
            xaxis = np.linspace(L.min(), L.max(), 5000)
            plt.plot(xaxis, kde(xaxis), label="L normal")
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.legend()

            kde = stats.gaussian_kde(L_delta)
            xaxis_delta = np.linspace(L_delta.min(), L_delta.max(), 5000)
            plt.plot(xaxis_delta, kde(xaxis_delta), label="L-delta normal")
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.legend()

            kde = stats.gaussian_kde(L_t)
            xaxis_t = np.linspace(L_t.min(), L_t.max(), 5000)
            plt.plot(xaxis_t, kde(xaxis_t), label="L t-student")
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.legend()
            plt.grid(True)


            #CDF
            plt.subplot(212)
            p = np.arange(len(L)) / (len(L) - 1)
            plt.plot(sorted(L), p, label="Loss fitted by normal")
            plt.legend()

            p = np.arange(len(L_delta)) / (len(L_delta) - 1)
            plt.plot(sorted(L_delta), p, label="Linearized Loss fitted by normal")
            plt.xlabel("loss(unit = $1)")
            plt.legend()

            p = np.arange(len(L_t)) / (len(L_t) - 1)
            plt.plot(sorted(L_t), p, label="Loss fitted by t-student")
            plt.xlabel("loss(unit = $1)")
            plt.legend()
            plt.grid(True)
            #plt.show()
            plt.savefig('compare-48-' + str(i) + '.png', bbox_inches='tight')


    ## Back test

    def back_test(self):
        L_act = [-(self.V_t[self.index[i+1]] - self.V_t[self.index[i]]) for i in range(24, 72)]
        # Plot a histogram of actual loss
        plt.hist(L_act, normed=True)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.grid()
        plt.title('Histogram of Loss')
        plt.xlabel("loss(unit=$1)")
        plt.show()
        # plot realized losses
        plt.plot(range(24, 72), L_act)
        plt.title("Loss time series")
        plt.ylabel("loss(unit=$1)")
        plt.xlabel("time(space = 1month)")
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.show()
        # Table for showing actual monthly losses

        # rolling window
        for i in range(24, 72):
            L, L_delta, L_t = self.calibrate(i)
            ## Fitted by normal
            # Loss distribution
            plt.figure(100+i)
            # PDF
            plt.subplot(211)
            plt.title("L fitted by normal", y=1.08)
            plt.hist(L, normed=True)
            kde = stats.gaussian_kde(L)
            xaxis = np.linspace(L.min(), L.max(), 1000)
            plt.plot(xaxis, kde(xaxis), label="PDF")
            plt.axvline(x=L_act[24 - i], color="red")
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.legend()
            plt.grid(True)
            # CDF
            plt.subplot(212)
            p = np.arange(len(L)) / (len(L) - 1)
            plt.plot(sorted(L), p, label="CDF")
            plt.axvline(x=L_act[24 - i], color="red")
            plt.xlabel("loss(unit = $1)")
            plt.legend()
            plt.grid(True)
            plt.savefig('48-' + str(i) + '-1' + '.png', bbox_inches='tight')
            #plt.show()
            # Linearized Loss distribution
            plt.figure(200+i)
            # PDF
            plt.subplot(211)
            plt.title("Linearized L fitted by normal", y=1.08)
            plt.hist(L_delta, normed=True)
            kde = stats.gaussian_kde(L_delta)
            xaxis_delta = np.linspace(L_delta.min(), L_delta.max(), 1000)
            plt.plot(xaxis_delta, kde(xaxis_delta), label="PDF")
            plt.axvline(x=L_act[24 - i], color="red")
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.legend()
            plt.grid(True)
            # CDF
            p = np.arange(len(L_delta)) / (len(L_delta) - 1)
            plt.subplot(212)
            plt.plot(sorted(L_delta), p, label="CDF")
            plt.axvline(x=L_act[24 - i], color="red")
            plt.xlabel("loss(unit = $1)")
            plt.legend()
            plt.grid(True)
            plt.savefig('48-' + str(i) + '-2' + '.png', bbox_inches='tight')
            #plt.show()

            ## Fitted by t-distribution
            plt.figure(300+i)
            # PDF
            plt.subplot(211)
            plt.title("L fitted by student-t", y=1.08)
            plt.hist(L_t, normed=True)
            kde = stats.gaussian_kde(L_t)
            xaxis_t = np.linspace(L_t.min(), L_t.max(), 1000)
            plt.plot(xaxis_t, kde(xaxis_t), label="PDF")
            plt.axvline(x=L_act[24 - i], color="red")
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.legend()
            plt.grid(True)
            # CDF
            p = np.arange(len(L_t)) / (len(L_t) - 1)
            plt.subplot(212)
            plt.plot(sorted(L_t), p, label="CDF")
            plt.axvline(x=L_act[24 - i], color="red")
            plt.xlabel("loss(unit = $1)")
            plt.legend()
            plt.grid(True)
            plt.savefig('48-' + str(i) + '-3' + '.png', bbox_inches='tight')
            plt.show()


    ## Risk

    def risk_between_normals_student(self):
        # Loss L
        L_VaR_95 = []
        L_CVaR_95 = []
        L_VaR_99 = []
        L_CVaR_99 = []
        # Linearized loss L_delta
        L_delta_VaR_95 = []
        L_delta_CVaR_95 = []
        L_delta_VaR_99 = []
        L_delta_CVaR_99 = []
        # T-student loss L_t
        L_t_VaR_95 = []
        L_t_CVaR_95 = []
        L_t_VaR_99 = []
        L_t_CVaR_99 = []
        for i in range(24, 72):
            print("It's time " + str(i))
            L, L_delta, L_t = self.calibrate(i)
            # Normal case
            # For Loss L
            L_fit_mu, L_fit_sigma = norm.fit(L)
            # Calculate VaR and CVaR
            L_VaR_95.append(norm.ppf(0.95, L_fit_mu, L_fit_sigma))
            L_CVaR_95.append(L_fit_mu + L_fit_sigma * (norm.pdf(norm.ppf(0.95)) / (1 - 0.95)))
            L_VaR_99.append(norm.ppf(0.99, L_fit_mu, L_fit_sigma))
            L_CVaR_99.append(L_fit_mu + L_fit_sigma * (norm.pdf(norm.ppf(0.99)) / (1 - 0.99)))

            # For linearized Loss L_delta
            L_delta_fit_mu, L_delta_fit_sigma = norm.fit(L_delta)
            # Calculate VaR and CVaR
            L_delta_VaR_95.append(norm.ppf(0.95, L_delta_fit_mu, L_delta_fit_sigma))
            L_delta_CVaR_95.append(L_delta_fit_mu + L_delta_fit_sigma * (norm.pdf(norm.ppf(0.95)) / (1 - 0.95)))
            L_delta_VaR_99.append(norm.ppf(0.99, L_delta_fit_mu, L_delta_fit_sigma))
            L_delta_CVaR_99.append(L_delta_fit_mu + L_delta_fit_sigma * (norm.pdf(norm.ppf(0.99)) / (1 - 0.99)))

            # T-student case
            # For Loss L
            while True:
                L_t_fit_df, L_t_fit_mu, L_t_fit_sigma = t.fit(L_t)
                print(L_t_fit_df)
                if L_t_fit_df > 2:
                    break
                L, L_delta, L_t = self.calibrate(i)
            # Calculate VaR and CVaR
            L_t_VaR_95.append(t.ppf(0.95, L_t_fit_df, L_t_fit_mu, L_t_fit_sigma))
            L_t_CVaR_95.append(L_fit_mu + L_fit_sigma * ((t.pdf(t.ppf(0.95, L_t_fit_df), L_t_fit_df) / (1 - 0.95))) * ( ((L_t_fit_df + (t.ppf(0.95, L_t_fit_df))**2) / (L_t_fit_df - 1) )) )
            L_t_VaR_99.append(t.ppf(0.99, L_t_fit_df, L_t_fit_mu, L_t_fit_sigma))
            L_t_CVaR_99.append( L_fit_mu + L_fit_sigma * ((t.pdf(t.ppf(0.99, L_t_fit_df), L_t_fit_df) / (1 - 0.99))) * ( ((L_t_fit_df + (t.ppf(0.99, L_t_fit_df))**2) / (L_t_fit_df - 1) )) )
        # Plot
        # For 0.95 case
        plt.plot(range(24, 72), L_VaR_95, label='VaR_.95 Loss')
        plt.legend()
        plt.plot(range(24, 72), L_CVaR_95, label='CVaR_.95 Loss')
        plt.legend()
        plt.plot(range(24, 72), L_delta_VaR_95, label='VaR_.95 Linearized Loss')
        plt.legend()
        plt.plot(range(24, 72), L_delta_CVaR_95, label='CVaR_.95 Linearized Loss')
        plt.legend()
        plt.plot(range(24, 72), L_t_VaR_95, label='VaR_.95 Loss-t')
        plt.legend()
        plt.plot(range(24, 72), L_t_CVaR_95, label='CVaR_.95 Loss-t')
        plt.legend()
        plt.show()
        # For 0.99 case
        plt.plot(range(24, 72), L_VaR_99, label='VaR_.99 Loss')
        plt.legend()
        plt.plot(range(24, 72), L_CVaR_99, label='CVaR_.99 Loss')
        plt.legend()
        plt.plot(range(24, 72), L_delta_VaR_99, label='VaR_.99 Linearized Loss')
        plt.legend()
        plt.plot(range(24, 72), L_delta_CVaR_99, label='CVaR_.99 Linearized Loss')
        plt.legend()
        plt.plot(range(24, 72), L_t_VaR_99, label='VaR_.99 Loss-t')
        plt.legend()
        plt.plot(range(24, 72), L_t_CVaR_99, label='CVaR_.99 Loss-t')
        plt.legend()
        plt.show()

        # Save these VaR and CVaR as .txt
        with open("test.txt", "wb") as fp:  # Pickling
            pickle.dump([[L_VaR_95, L_CVaR_95, L_delta_VaR_95, L_delta_CVaR_95, L_t_VaR_95, L_t_CVaR_95],
                         [L_VaR_99, L_CVaR_99, L_delta_VaR_99, L_delta_CVaR_99, L_t_VaR_99, L_t_CVaR_99]], fp)

        ## Difference among actual monthly losses and VaR & CVaR values
        with open("test.txt", "rb") as fp:  # Unpickling
            VaR_CVaR_list = pickle.load(fp)
        xaxis = range(24, 72)
        L_act = [-(self.V_t[self.index[i + 1]] - self.V_t[self.index[i]]) for i in range(24, 72)]
        plt.subplot(311)
        plt.plot(range(24, 72), (np.asarray(VaR_CVaR_list[0][0]) - L_act) / np.asarray(VaR_CVaR_list[0][0]),
                 label='VaR_.95 Loss')
        plt.legend()
        plt.subplot(312)
        plt.plot(range(24, 72), (np.asarray(VaR_CVaR_list[0][2]) - L_act) / np.asarray(VaR_CVaR_list[0][2]),
                 label='VaR_.95 Linearized Loss')
        plt.legend()
        plt.subplot(313)
        plt.plot(range(24, 72), (np.asarray(VaR_CVaR_list[0][4]) - L_act) / np.asarray(VaR_CVaR_list[0][4]),
                 label='VaR_.95 Loss-t')
        plt.legend()
        plt.xlabel("time(space = 1month)")
        plt.ylabel('percentage')
        plt.show()
        plt.subplot(311)
        plt.plot(range(24, 72), (np.asarray(VaR_CVaR_list[0][1]) - L_act) / np.asarray(VaR_CVaR_list[0][1]),
                 label='CVaR_.95 Loss')
        plt.legend()
        plt.subplot(312)
        plt.plot(range(24, 72), (np.asarray(VaR_CVaR_list[0][3]) - L_act) / np.asarray(VaR_CVaR_list[0][3]),
                 label='CVaR_.95 Linearized Loss')
        plt.legend()
        plt.subplot(313)
        plt.plot(range(24, 72), (np.asarray(VaR_CVaR_list[0][5]) - L_act) / np.asarray(VaR_CVaR_list[0][5]),
                 label='CVaR_.95 Loss-t')
        plt.legend()
        plt.xlabel("time(space = 1month)")
        plt.ylabel('percentage')
        plt.show()

        plt.subplot(311)
        plt.plot(range(24, 72), (np.asarray(VaR_CVaR_list[1][0]) - L_act) / np.asarray(VaR_CVaR_list[1][0]),
                 label='VaR_.99 Loss')
        plt.legend()
        plt.subplot(312)
        plt.plot(range(24, 72), (np.asarray(VaR_CVaR_list[1][2]) - L_act) / np.asarray(VaR_CVaR_list[1][2]),
                 label='VaR_.99 Linearized Loss')
        plt.legend()
        plt.subplot(313)
        plt.plot(range(24, 72), (np.asarray(VaR_CVaR_list[1][4]) - L_act) / np.asarray(VaR_CVaR_list[1][4]),
                 label='VaR_.99 Loss-t')
        plt.legend()
        plt.xlabel("time(space = 1month)")
        plt.ylabel('percentage')
        plt.show()
        plt.subplot(311)
        plt.plot(range(24, 72), (np.asarray(VaR_CVaR_list[1][1]) - L_act) / np.asarray(VaR_CVaR_list[1][1]),
                 label='CVaR_.99 Loss')
        plt.legend()
        plt.subplot(312)
        plt.plot(range(24, 72), (np.asarray(VaR_CVaR_list[1][3]) - L_act) / np.asarray(VaR_CVaR_list[1][3]),
                 label='CVaR_.99 Linearized Loss')
        plt.legend()
        plt.subplot(313)
        plt.plot(range(24, 72), (np.asarray(VaR_CVaR_list[1][5]) - L_act) / np.asarray(VaR_CVaR_list[1][5]),
                 label='CVaR_.99 Loss-t')
        plt.legend()
        plt.xlabel("time(space = 1month)")
        plt.ylabel('percentage')
        plt.show()

    def regression_estimator(self, week_num, show_figure):
        #index_12_30_16 = np.nonzero(self.weekly_date == '12/30/16')[0][0]
        index_1_2_15 = np.nonzero(self.weekly_date == '1/2/15')[0][0]
        L_act = [-(self.V_t[self.weekly_index[i + 1]] - self.V_t[self.weekly_index[i]]) for i in
                    range(index_1_2_15 + week_num - 140, index_1_2_15 + week_num - 1)]
        sorted_L_act = sorted(L_act)
        sorted_L_act = np.asarray(sorted_L_act)
        ## K is corresponding with VaR_alpha, we try starting with 0.8 to a number around 1, to make sure L_k is larger than 0.
        emp_VaR_k = 0
        k = 0.79
        while emp_VaR_k <= 0:
            k += 0.1
            emp_VaR_k = sorted_L_act[round(k * len(sorted_L_act)) - 1]
            if k >= 1:
                print("This period historical monthly losses all are negative number.")
                break
        right_tail_L_emp_k = sorted_L_act[sorted_L_act >= emp_VaR_k]
        x = []
        for i in range(len(right_tail_L_emp_k)):
            x.append(np.log((len(right_tail_L_emp_k) - i) / len(right_tail_L_emp_k)))
        x = np.asarray(x)
        Y = np.log(right_tail_L_emp_k)
        Y = Y.reshape(len(Y), 1)
        X = x.reshape(len(x), 1)
        X = np.concatenate((np.ones([len(X), 1]), X), axis=1)
        Beta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), Y)
        a = - 1/Beta[1][0]
        A = - (np.exp(-Beta[0][0]/Beta[1][0])/Beta[1][0])
        if show_figure == "Yes":
            Xaxis = []
            Yaxis = []
            for i in range(0, len(right_tail_L_emp_k)):
                Xaxis.append(np.log((len(right_tail_L_emp_k) - i) / len(right_tail_L_emp_k)))
            Xaxis = np.asarray(Xaxis)
            Yaxis = Y
            plt.scatter(Xaxis, Yaxis)
            plt.plot(Xaxis, Beta[0][0] + Beta[1][0] * Xaxis, color='red')
            plt.xlabel("Log((n-k)/n)")
            plt.ylabel('Log(L_(k))')
            plt.title("Linearized Regression Plot")
            plt.show()
        # Calculate VaR and ES
        VaR_90 = sorted_L_act[round(0.9 * len(sorted_L_act)) - 1]
        VaR_95 = VaR_90 * ((1 - 0.9) / (1 - 0.95))**(1 / a)
        VaR_99 = VaR_90 * ((1 - 0.9) / (1 - 0.99))**(1 / a)
        ES_95 = VaR_95 * (a / (a - 1))
        ES_99 = VaR_99 * (a / (a - 1))
        return a, VaR_95, VaR_99, ES_95, ES_99

    def hill_estimator(self, week_num, show_figure):
        #index_12_30_16 = np.nonzero(self.weekly_date == '12/30/16')[0][0]
        index_1_2_15 = np.nonzero(self.weekly_date == '1/2/15')[0][0]
        L_act = [-(self.V_t[self.weekly_index[i + 1]] - self.V_t[self.weekly_index[i]]) for i in
                 range(index_1_2_15 + week_num - 140, index_1_2_15 + week_num - 1)]
        #L_act = [-(self.V_t[self.index[i + 1]] - self.V_t[self.index[i]]) for i in range(month_num - 36, month_num)]
        sorted_L_act = sorted(L_act)
        upper_sorted_L_act = sorted_L_act[::-1]
        upper_sorted_L_act = np.asarray(upper_sorted_L_act)
        a_c = []
        for i in range(1, len(upper_sorted_L_act)):
            if upper_sorted_L_act[i] <= 0:
                break
            larger_than_C_L_act = upper_sorted_L_act[:i]
            a_c.append(len(larger_than_C_L_act)/sum(np.log(larger_than_C_L_act / upper_sorted_L_act[i])))
        if show_figure == "Yes":
            plt.scatter(range(len(a_c)), a_c)
            plt.ylim(0, 5)
            plt.xlabel("n(c)")
            plt.ylabel('a(c)')
            plt.title("Hill Plot")
            plt.show()
        # estimate a
        a_c = np.asarray(a_c)
        potential_a = []
        for i in range(len(a_c)):
            temp = 0
            differences = np.abs(a_c[i] - a_c)
            for j in range(len(differences)):
                if differences[j] <= 0.15:
                    temp += 1
            potential_a.append(temp)
        potential_a = np.asarray(potential_a)
        a = a_c[np.nonzero(potential_a == max(potential_a))[0][0]]
        # Calculate VaR and ES
        VaR_90 = sorted_L_act[round(0.9 * len(sorted_L_act)) - 1]
        VaR_95 = VaR_90 * ((1 - 0.9) / (1 - 0.95)) ** (1 / a)
        VaR_99 = VaR_90 * ((1 - 0.9) / (1 - 0.99)) ** (1 / a)
        ES_95 = VaR_95 * (a / (a - 1))
        ES_99 = VaR_99 * (a / (a - 1))
        return a, VaR_95, VaR_99, ES_95, ES_99

    def compare_regression_hill_a(self):
        final_friday_index = np.nonzero(self.weekly_date == '12/29/17')[0][0]
        index_12_30_16 = np.nonzero(self.weekly_date == '12/30/16')[0][0]
        last_year_range = range(final_friday_index - index_12_30_16)
        # Regression estimator
        Yaxis_regression = []
        for i in last_year_range:
            Yaxis_regression.append(self.regression_estimator(i, "No")[0])
        # Hill estimator
        Yaxis_hill = []
        for i in last_year_range:
            Yaxis_hill.append(self.hill_estimator(i, "No")[0])
        plt.plot(range(len(Yaxis_regression)), Yaxis_regression, label='Regression')
        plt.legend()
        plt.plot(range(len(Yaxis_hill)), Yaxis_hill, label='Hill')
        plt.legend()
        plt.title("Values of parameter a")
        plt.xlabel("Weeks")
        plt.ylabel("a")
        plt.show()

    def compare_regression_hill_VaR_ES(self):
        final_friday_index = np.nonzero(self.weekly_date == '12/29/17')[0][0]
        index_12_30_16 = np.nonzero(self.weekly_date == '12/30/16')[0][0]
        last_year_range = range(final_friday_index - index_12_30_16)
        # Regression estimator
        Yaxis_VaR_regression_95 = []
        Yaxis_VaR_regression_99 = []
        Yaxis_ES_regression_95 = []
        Yaxis_ES_regression_99 = []
        for i in last_year_range:
            Yaxis_VaR_regression_95.append(self.regression_estimator(i, "No")[1])
            Yaxis_VaR_regression_99.append(self.regression_estimator(i, "No")[2])
            Yaxis_ES_regression_95.append(self.regression_estimator(i, "No")[3])
            Yaxis_ES_regression_99.append(self.regression_estimator(i, "No")[4])
        # Hill estimator
        Yaxis_VaR_hill_95 = []
        Yaxis_VaR_hill_99 = []
        Yaxis_ES_hill_95 = []
        Yaxis_ES_hill_99 = []
        for i in last_year_range:
            Yaxis_VaR_hill_95.append(self.hill_estimator(i, "No")[1])
            Yaxis_VaR_hill_99.append(self.hill_estimator(i, "No")[2])
            Yaxis_ES_hill_95.append(self.hill_estimator(i, "No")[3])
            Yaxis_ES_hill_99.append(self.hill_estimator(i, "No")[4])
        # VaR
        plt.figure(1000)
        plt.plot(range(len(Yaxis_VaR_regression_95)), Yaxis_VaR_regression_95, label='Regression')
        plt.legend()
        plt.plot(range(len(Yaxis_VaR_hill_95)), Yaxis_VaR_hill_95, label='Hill')
        plt.legend()
        plt.title("Comparision of VaR between Regression and Hill at level 0.95")
        plt.xlabel("Weeks")
        plt.ylabel("VaR")
        plt.show()
        plt.figure(1001)
        plt.plot(range(len(Yaxis_VaR_regression_99)), Yaxis_VaR_regression_99, label='Regression')
        plt.legend()
        plt.plot(range(len(Yaxis_VaR_hill_99)), Yaxis_VaR_hill_99, label='Hill')
        plt.legend()
        plt.title("comparision of VaR between Regression and Hill at level 0.99")
        plt.xlabel("Weeks")
        plt.ylabel("VaR")
        plt.show()
        # ES
        plt.figure(1003)
        plt.plot(range(len(Yaxis_ES_regression_95)), Yaxis_ES_regression_95, label='Regression')
        plt.legend()
        plt.plot(range(len(Yaxis_ES_hill_95)), Yaxis_ES_hill_95, label='Hill')
        plt.legend()
        plt.title("comparision of ES between Regression and Hill at level 0.95")
        plt.xlabel("Weeks")
        plt.ylabel("VaR")
        plt.show()
        plt.figure(1004)
        plt.plot(range(len(Yaxis_ES_regression_99)), Yaxis_ES_regression_99, label='Regression')
        plt.legend()
        plt.plot(range(len(Yaxis_ES_hill_99)), Yaxis_ES_hill_99, label='Hill')
        plt.legend()
        plt.title("comparision of ES between Regression and Hill at level 0.99")
        plt.xlabel("Weeks")
        plt.ylabel("VaR")
        plt.show()

    def risk_among_5_models(self):
        L_act = np.asarray([-(self.V_t[self.index[i + 1]] - self.V_t[self.index[i]]) for i in range(36, 56)])
        final_friday_index = np.nonzero(self.weekly_date == '7/15/16')[0][0]
        index_1_2_15 = np.nonzero(self.weekly_date == '1/2/15')[0][0]
        last_year_range = range(0, final_friday_index - index_1_2_15, 4)
        print(len(last_year_range))
        # Loss L
        L_VaR_95 = []
        L_CVaR_95 = []
        L_VaR_99 = []
        L_CVaR_99 = []
        # Linearized loss L_delta
        L_delta_VaR_95 = []
        L_delta_CVaR_95 = []
        L_delta_VaR_99 = []
        L_delta_CVaR_99 = []
        # T-studen loss L_t
        L_t_VaR_95 = []
        L_t_CVaR_95 = []
        L_t_VaR_99 = []
        L_t_CVaR_99 = []
        # Regression estimator
        Yaxis_VaR_regression_95 = []
        Yaxis_VaR_regression_99 = []
        Yaxis_ES_regression_95 = []
        Yaxis_ES_regression_99 = []
        for i in last_year_range:
            Yaxis_VaR_regression_95.append(self.regression_estimator(i, "No")[1])
            Yaxis_VaR_regression_99.append(self.regression_estimator(i, "No")[2])
            Yaxis_ES_regression_95.append(self.regression_estimator(i, "No")[3])
            Yaxis_ES_regression_99.append(self.regression_estimator(i, "No")[4])
        # Hill estimator
        Yaxis_VaR_hill_95 = []
        Yaxis_VaR_hill_99 = []
        Yaxis_ES_hill_95 = []
        Yaxis_ES_hill_99 = []
        for i in last_year_range:
            Yaxis_VaR_hill_95.append(self.hill_estimator(i, "No")[1])
            Yaxis_VaR_hill_99.append(self.hill_estimator(i, "No")[2])
            Yaxis_ES_hill_95.append(self.hill_estimator(i, "No")[3])
            Yaxis_ES_hill_99.append(self.hill_estimator(i, "No")[4])
        for i in range(36, 56):
            print("It's time " + str(i))
            L, L_delta, L_t = self.calibrate(i)
            # Normal case
            # For Loss L
            L_fit_mu, L_fit_sigma = norm.fit(L)
            # Calculate VaR and CVaR
            L_VaR_95.append(norm.ppf(0.95, L_fit_mu, L_fit_sigma))
            L_CVaR_95.append(L_fit_mu + L_fit_sigma * (norm.pdf(norm.ppf(0.95)) / (1 - 0.95)))
            L_VaR_99.append(norm.ppf(0.99, L_fit_mu, L_fit_sigma))
            L_CVaR_99.append(L_fit_mu + L_fit_sigma * (norm.pdf(norm.ppf(0.99)) / (1 - 0.99)))

            # For linearized Loss L_delta
            L_delta_fit_mu, L_delta_fit_sigma = norm.fit(L_delta)
            # Calculate VaR and CVaR
            L_delta_VaR_95.append(norm.ppf(0.95, L_delta_fit_mu, L_delta_fit_sigma))
            L_delta_CVaR_95.append(L_delta_fit_mu + L_delta_fit_sigma * (norm.pdf(norm.ppf(0.95)) / (1 - 0.95)))
            L_delta_VaR_99.append(norm.ppf(0.99, L_delta_fit_mu, L_delta_fit_sigma))
            L_delta_CVaR_99.append(L_delta_fit_mu + L_delta_fit_sigma * (norm.pdf(norm.ppf(0.99)) / (1 - 0.99)))

            # T-student case
            # For Loss L
            while True:
                L_t_fit_df, L_t_fit_mu, L_t_fit_sigma = t.fit(L_t)
                print(L_t_fit_df)
                if L_t_fit_df > 2:
                    break
                L, L_delta, L_t = self.calibrate(i)
            # Calculate VaR and CVaR
            L_t_VaR_95.append(t.ppf(0.95, L_t_fit_df, L_t_fit_mu, L_t_fit_sigma))
            L_t_CVaR_95.append(L_fit_mu + L_fit_sigma * ((t.pdf(t.ppf(0.95, L_t_fit_df), L_t_fit_df) / (1 - 0.95))) * (
            ((L_t_fit_df + (t.ppf(0.95, L_t_fit_df)) ** 2) / (L_t_fit_df - 1))))
            L_t_VaR_99.append(t.ppf(0.99, L_t_fit_df, L_t_fit_mu, L_t_fit_sigma))
            L_t_CVaR_99.append(L_fit_mu + L_fit_sigma * ((t.pdf(t.ppf(0.99, L_t_fit_df), L_t_fit_df) / (1 - 0.99))) * (
            ((L_t_fit_df + (t.ppf(0.99, L_t_fit_df)) ** 2) / (L_t_fit_df - 1))))
        # Plot
        #For 0.95 case
        plt.subplot(121)
        plt.plot(range(36, 56), L_VaR_95, label='VaR_.95 Loss')
        plt.legend()
        plt.plot(range(36, 56), L_CVaR_95, label='CVaR_.95 Loss')
        plt.legend()
        plt.plot(range(36, 56), L_delta_VaR_95, label='VaR_.95 Linearized Loss')
        plt.legend()
        plt.plot(range(36, 56), L_delta_CVaR_95, label='CVaR_.95 Linearized Loss')
        plt.legend()
        plt.plot(range(36, 56), L_t_VaR_95, label='VaR_.95 Loss-t')
        plt.legend()
        plt.plot(range(36, 56), L_t_CVaR_95, label='CVaR_.95 Loss-t')
        plt.legend()
        plt.plot(range(36, 56), Yaxis_VaR_regression_95, label='VaR_.95 Regression')
        plt.legend()
        plt.plot(range(36, 56), Yaxis_ES_regression_95, label='CVaR_.95 Regression')
        plt.legend()
        plt.plot(range(36, 56), Yaxis_VaR_hill_95, label='VaR_.95 Hill')
        plt.legend()
        plt.plot(range(36, 56), Yaxis_ES_hill_95, label='CVaR_.95 Hill')
        plt.legend()

        plt.subplot(122)
        plt.plot(range(36, 56), L_VaR_95, label='VaR_.95 Loss')
        plt.legend()
        plt.plot(range(36, 56), L_CVaR_95, label='CVaR_.95 Loss')
        plt.legend()
        plt.plot(range(36, 56), L_delta_VaR_95, label='VaR_.95 Linearized Loss')
        plt.legend()
        plt.plot(range(36, 56), L_delta_CVaR_95, label='CVaR_.95 Linearized Loss')
        plt.legend()
        plt.plot(range(36, 56), L_t_VaR_95, label='VaR_.95 Loss-t')
        plt.legend()
        plt.plot(range(36, 56), L_t_CVaR_95, label='CVaR_.95 Loss-t')
        plt.legend()
        plt.plot(range(36, 56), Yaxis_VaR_regression_95, label='VaR_.95 Regression')
        plt.legend()
        plt.plot(range(36, 56), Yaxis_ES_regression_95, label='CVaR_.95 Regression')
        plt.legend()
        plt.plot(range(36, 56), Yaxis_VaR_hill_95, label='VaR_.95 Hill')
        plt.legend()
        plt.plot(range(36, 56), Yaxis_ES_hill_95, label='CVaR_.95 Hill')
        plt.legend()
        plt.ylim(0, 300000)
        plt.show()
        # For 0.99 case
        plt.subplot(121)
        plt.plot(range(36, 56), L_VaR_99, label='VaR_.99 Loss')
        plt.legend()
        plt.plot(range(36, 56), L_CVaR_99, label='CVaR_.99 Loss')
        plt.legend()
        plt.plot(range(36, 56), L_delta_VaR_99, label='VaR_.99 Linearized Loss')
        plt.legend()
        plt.plot(range(36, 56), L_delta_CVaR_99, label='CVaR_.99 Linearized Loss')
        plt.legend()
        plt.plot(range(36, 56), L_t_VaR_99, label='VaR_.99 Loss-t')
        plt.legend()
        plt.plot(range(36, 56), L_t_CVaR_99, label='CVaR_.99 Loss-t')
        plt.legend()
        plt.plot(range(36, 56), Yaxis_VaR_regression_99, label='VaR_.99 Regression')
        plt.legend()
        plt.plot(range(36, 56), Yaxis_ES_regression_99, label='CVaR_.99 Regression')
        plt.legend()
        plt.plot(range(36, 56), Yaxis_VaR_hill_99, label='VaR_.99 Hill')
        plt.legend()
        plt.plot(range(36, 56), Yaxis_ES_hill_99, label='CVaR_.99 Hill')
        plt.legend()

        plt.subplot(122)
        plt.plot(range(36, 56), L_VaR_99, label='VaR_.99 Loss')
        plt.legend()
        plt.plot(range(36, 56), L_CVaR_99, label='CVaR_.99 Loss')
        plt.legend()
        plt.plot(range(36, 56), L_delta_VaR_99, label='VaR_.99 Linearized Loss')
        plt.legend()
        plt.plot(range(36, 56), L_delta_CVaR_99, label='CVaR_.99 Linearized Loss')
        plt.legend()
        plt.plot(range(36, 56), L_t_VaR_99, label='VaR_.99 Loss-t')
        plt.legend()
        plt.plot(range(36, 56), L_t_CVaR_99, label='CVaR_.99 Loss-t')
        plt.legend()
        plt.plot(range(36, 56), Yaxis_VaR_regression_99, label='VaR_.99 Regression')
        plt.legend()
        plt.plot(range(36, 56), Yaxis_ES_regression_99, label='CVaR_.99 Regression')
        plt.legend()
        plt.plot(range(36, 56), Yaxis_VaR_hill_99, label='VaR_.99 Hill')
        plt.legend()
        plt.plot(range(36, 56), Yaxis_ES_hill_99, label='CVaR_.99 Hill')
        plt.legend()
        plt.ylim(0, 600000)
        plt.show()
        plt.show()

        ## Difference between actual monthly losses and VaR & CVaR values
        xaxis = range(36, 56)
        L_act = np.asarray([-(self.V_t[self.index[i + 1]] - self.V_t[self.index[i]]) for i in range(36, 56)])
        last_year__act_range = range(index_1_2_15, final_friday_index, 4)
        L_weekly_act = np.asarray(
            [-(self.V_t[self.weekly_index[i + 1]] - self.V_t[self.weekly_index[i]]) for i in last_year__act_range])
        # VaR.95
        plt.subplot(321)
        plt.plot(range(36, 56), (np.asarray(L_VaR_95) - L_act) / np.asarray(L_VaR_95),
                 label='VaR_.95 Loss')
        plt.legend()
        plt.subplot(322)
        plt.plot(range(36, 56), (np.asarray(L_delta_VaR_95) - L_act) / np.asarray(L_delta_VaR_95),
                 label='VaR_.95 Linearized Loss')
        plt.legend()
        plt.subplot(323)
        plt.plot(range(36, 56), (np.asarray(L_t_VaR_95) - L_act) / np.asarray(L_t_VaR_95),
                 label='VaR_.95 Loss-t')
        plt.legend()
        plt.subplot(324)
        plt.plot(range(36, 56), (np.asarray(Yaxis_VaR_regression_95) - L_weekly_act) / np.asarray(Yaxis_VaR_regression_95),
                 label='VaR_.95 Regression')
        plt.legend()
        plt.subplot(325)
        plt.plot(range(36, 56), (np.asarray(Yaxis_VaR_hill_95) - L_weekly_act) / np.asarray(Yaxis_VaR_hill_95),
                 label='VaR_.95 Hill')
        plt.legend()
        plt.ylim(-1, 10)
        plt.xlabel("time(space = 1month)")
        plt.ylabel('percentage')
        plt.show()
        # VaR_099
        plt.subplot(321)
        plt.plot(range(36, 56), (np.asarray(L_VaR_99) - L_act) / np.asarray(L_VaR_99),
                 label='VaR_.99 Loss')
        plt.legend()
        plt.subplot(322)
        plt.plot(range(36, 56), (np.asarray(L_delta_VaR_99) - L_act) / np.asarray(L_delta_VaR_99),
                 label='VaR_.99 Linearized Loss')
        plt.legend()
        plt.subplot(323)
        plt.plot(range(36, 56), (np.asarray(L_t_VaR_99) - L_act) / np.asarray(L_t_VaR_99),
                 label='VaR_.99 Loss-t')
        plt.legend()
        plt.subplot(324)
        plt.plot(range(36, 56), (np.asarray(Yaxis_VaR_regression_99) - L_weekly_act) / np.asarray(Yaxis_VaR_regression_99),
                 label='VaR_.99 Regression')
        plt.legend()
        plt.subplot(325)
        plt.plot(range(36, 56), (np.asarray(Yaxis_VaR_hill_99) - L_weekly_act) / np.asarray(Yaxis_VaR_hill_99),
                 label='VaR_.99 Hill')
        plt.legend()
        plt.ylim(-1, 10)
        plt.xlabel("time(space = 1month)")
        plt.ylabel('percentage')
        plt.show()
        # ES.95
        plt.subplot(321)
        plt.plot(range(36, 56), (np.asarray(L_CVaR_95) - L_act) / np.asarray(L_CVaR_95),
                 label='CVaR_.95 Loss')
        plt.legend()
        plt.subplot(322)
        plt.plot(range(36, 56), (np.asarray(L_delta_CVaR_95) - L_act) / np.asarray(L_delta_CVaR_95),
                 label='CVaR_.95 Linearized Loss')
        plt.legend()
        plt.subplot(323)
        plt.plot(range(36, 56), (np.asarray(L_t_CVaR_95) - L_act) / np.asarray(L_t_CVaR_95),
                 label='CVaR_.95 Loss-t')
        plt.legend()
        plt.subplot(324)
        plt.plot(range(36, 56), (np.asarray(Yaxis_ES_regression_95) - L_weekly_act) / np.asarray(Yaxis_ES_regression_95),
                 label='CVaR_.95 Regression')
        plt.legend()
        plt.subplot(325)
        plt.plot(range(36, 56), (np.asarray(Yaxis_ES_hill_95) - L_weekly_act) / np.asarray(Yaxis_ES_hill_95),
                 label='CVaR_.95 Hill')
        plt.legend()
        plt.ylim(-1, 10)
        plt.xlabel("time(space = 1month)")
        plt.ylabel('percentage')
        plt.show()
        #ES.99
        plt.subplot(321)
        plt.plot(range(36, 56), (np.asarray(L_CVaR_99) - L_act) / np.asarray(L_CVaR_99),
                 label='CVaR_.99 Loss')
        plt.legend()
        plt.subplot(322)
        plt.plot(range(36, 56), (np.asarray(L_delta_CVaR_99) - L_act) / np.asarray(L_delta_CVaR_99),
                 label='CVaR_.99 Linearized Loss')
        plt.legend()
        plt.subplot(323)
        plt.plot(range(36, 56), (np.asarray(L_t_CVaR_99) - L_act) / np.asarray(L_t_CVaR_99),
                 label='CVaR_.99 Loss-t')
        plt.legend()
        plt.subplot(324)
        plt.plot(range(36, 56), (np.asarray(Yaxis_ES_regression_99) - L_weekly_act) / np.asarray(Yaxis_ES_regression_99),
                 label='CVaR_.99 Regression')
        plt.legend()
        plt.subplot(325)
        plt.plot(range(36, 56), (np.asarray(Yaxis_ES_hill_99) - L_weekly_act) / np.asarray(Yaxis_ES_hill_99),
                 label='CVaR_.99 Hill')
        plt.legend()
        plt.ylim(-1, 10)
        plt.xlabel("time(space = 1month)")
        plt.ylabel('percentage')
        plt.show()

    def get_ARMA_GARCH_parameters(self, week_num, dist_):
        self.cal_weekly_V_t(week_num)
        L_act = [-(self.V_t[i + 1] - self.V_t[i]) for i in range(len(self.V_t) - 1)]
        L_act = np.asarray(L_act)
        arma = smt.ARMA(L_act-np.mean(L_act), order=(1, 1)).fit(start_params=(0.1, 0.1), method='mle', trend='nc')
        #print(arma.summary())
        garch = arch_model(arma.resid, mean='Zero', p=1, o=0, q=1, dist=dist_).fit(update_freq=5, disp='off')
        #print(garch.summary())
        return L_act, np.mean(L_act), arma.params, garch.params

    def get_GARCH_mu_sigma_x(self, week_num):
        # GARCH model with normal
        GARCH_normal_params = self.get_ARMA_GARCH_parameters(week_num, 'Normal')
        mu = GARCH_normal_params[1]
        mu_t_normal = [GARCH_normal_params[0][0]]
        sigma_sq_t_normal = [0]
        x_t_normal = [GARCH_normal_params[0][0]]
        for i in range(1, len(GARCH_normal_params[0]) + 1):
            mu_t_normal.append(mu + GARCH_normal_params[2][0]*(x_t_normal[i-1] - mu) + GARCH_normal_params[2][1]*(x_t_normal[i-1] - mu_t_normal[i-1]))
            sigma_sq_t_normal.append(GARCH_normal_params[3][0]
                                     + GARCH_normal_params[3][1]*((x_t_normal[i-1] - mu_t_normal[i-1])**2)
                                     + GARCH_normal_params[3][2]*sigma_sq_t_normal[i-1])
            x_t_normal.append(mu_t_normal[-1] + np.sqrt(sigma_sq_t_normal[-1])*np.random.normal())
        # GARCH model with t-student
        GARCH_student_params = self.get_ARMA_GARCH_parameters(week_num, 'StudentsT')
        mu = GARCH_student_params[1]
        mu_t_student = [GARCH_student_params[0][0]]
        sigma_sq_t_student = [0]
        x_t_student = [GARCH_student_params[0][0]]
        for i in range(1, len(GARCH_student_params[0]) + 1):
            mu_t_student.append(mu + GARCH_student_params[2][0] * (x_t_student[i - 1] - mu) + GARCH_student_params[2][1] * (
                        x_t_student[i - 1] - mu_t_student[i - 1]))
            sigma_sq_t_student.append(GARCH_student_params[3][0]
                                     + GARCH_student_params[3][1] * ((x_t_student[i - 1] - mu_t_student[i - 1]) ** 2)
                                     + GARCH_student_params[3][2] * sigma_sq_t_student[i - 1])
            x_t_student.append(mu_t_normal[-1] + np.sqrt(sigma_sq_t_normal[-1]) * t.rvs(GARCH_student_params[3][3]))
        return mu_t_normal, sigma_sq_t_normal, x_t_normal, mu_t_student, sigma_sq_t_student, x_t_student

    def compare_GARCH_dists(self, week_num):
        self.cal_weekly_V_t(week_num)
        L_act = [-(self.V_t[i + 1] - self.V_t[i]) for i in range(len(self.V_t) - 1)]
        L_act = np.asarray(L_act)
        params = self.get_GARCH_mu_sigma_x(week_num)
        mu_t_normal = params[0]
        sigma_sq_t_normal = np.sqrt(params[1])
        x_t_normal = params[2]
        mu_t_student = params[3]
        sigma_sq_t_student = np.sqrt(params[4])
        x_t_student = params[5]

        plt.plot(range(len(mu_t_normal)), mu_t_normal, label="Normal")
        plt.plot(range(len(mu_t_student)), mu_t_student, label="Student")
        plt.legend()
        plt.title("Conditional mean between two models")
        plt.xlabel("Weeks")
        plt.ylabel("Values")
        plt.show()
        plt.plot(range(len(sigma_sq_t_normal)), np.sqrt(sigma_sq_t_normal), label="Normal")
        plt.plot(range(len(sigma_sq_t_student)), np.sqrt(sigma_sq_t_student), label="Student")
        plt.legend()
        plt.title("Conditional standard deviation between two models")
        plt.xlabel("Weeks")
        plt.ylabel("Values")
        plt.show()
        plt.plot(range(len(L_act)), L_act, label="Actual Loss")
        plt.plot(range(len(x_t_normal)), x_t_normal, label="Normal")
        plt.plot(range(len(x_t_student)), x_t_student, label="Student")
        plt.legend()
        plt.title("Losses among two models and actual loss")
        plt.xlabel("Weeks")
        plt.ylabel("Values")
        plt.show()

    def risk_between_GARCH(self):
        final_friday_index = np.nonzero(self.weekly_date == '12/29/17')[0][0]
        index_1_2_15 = np.nonzero(self.weekly_date == '1/2/15')[0][0]
        last_year_range = range(0, final_friday_index - index_1_2_15, 4)
        L_act = []
        # GARCH normal
        G_normal_VaR_95 = []
        G_normal_CVaR_95 = []
        G_normal_VaR_99 = []
        G_normal_CVaR_99 = []
        # GARCH student
        G_student_VaR_95 = []
        G_student_CVaR_95 = []
        G_student_VaR_99 = []
        G_student_CVaR_99 = []

        for i in last_year_range:
            #if i==53 and i==56:
            #    continue
            print("This is " + str(i))
            L_act.append(-(self.V_t[self.weekly_index[index_1_2_15 + i]] - self.V_t[self.weekly_index[index_1_2_15 + i - 1]]))
            params = self.get_GARCH_mu_sigma_x(i)
            mu_tplus1_normal = params[0][-1]
            sigma_tplus1_normal = np.sqrt(params[1][-1])
            mu_tplus1_student = params[3][-1]
            sigma_tplus1_student = np.sqrt(params[4][-1])
            df = self.get_ARMA_GARCH_parameters(i, "StudentsT")[3][3]

            G_normal_VaR_95.append(mu_tplus1_normal + sigma_tplus1_normal*norm.ppf(0.95))
            G_normal_CVaR_95.append(mu_tplus1_normal + sigma_tplus1_normal*(norm.pdf(norm.ppf(0.95)) / (1 - 0.95)))
            G_normal_VaR_99.append(mu_tplus1_normal + sigma_tplus1_normal * norm.ppf(0.99))
            G_normal_CVaR_99.append(mu_tplus1_normal + sigma_tplus1_normal * (norm.pdf(norm.ppf(0.99)) / (1 - 0.99)))

            G_student_VaR_95.append(mu_tplus1_student + sigma_tplus1_student * t.ppf(0.95, df))
            G_student_CVaR_95.append(mu_tplus1_student + sigma_tplus1_student
                                     * ((t.pdf(t.ppf(0.95, df), df) / (1 - 0.95))) * (((df + (t.ppf(0.95, df))**2) / (df - 1))))
            G_student_VaR_99.append(mu_tplus1_student + sigma_tplus1_student * t.ppf(0.99, df))
            G_student_CVaR_99.append(mu_tplus1_student + sigma_tplus1_student
                                     * ((t.pdf(t.ppf(0.99, df), df) / (1 - 0.99))) * (((df + (t.ppf(0.99, df))**2) / (df - 1))))
        plt.subplot(121)
        plt.plot(last_year_range, G_normal_VaR_95, label='Normal_VaR_0.95')
        plt.plot(last_year_range, G_student_VaR_95, label='Student_VaR_0.95')
        plt.plot(last_year_range, L_act, label='Actual Loss')
        plt.legend()
        plt.xlabel("Weeks")
        plt.ylabel("Values")

        plt.subplot(122)
        plt.plot(last_year_range, G_normal_CVaR_95, label='Normal_CVaR_0.95')
        plt.plot(last_year_range, G_student_CVaR_95, label='Student_CVaR_0.95')
        plt.plot(last_year_range, L_act, label='Actual Loss')
        plt.legend()
        plt.xlabel("Weeks")
        plt.suptitle("VaR_0.95 & CVaR_0.95 between two models versus time")
        plt.show()

        plt.subplot(121)
        plt.plot(last_year_range, G_normal_VaR_99, label='Normal_VaR_0.99')
        plt.plot(last_year_range, G_student_VaR_99, label='Student_VaR_0.99')
        plt.plot(last_year_range, L_act, label='Actual Loss')
        plt.legend()
        plt.xlabel("Weeks")
        plt.ylabel("Values")

        plt.subplot(122)
        plt.plot(last_year_range, G_normal_CVaR_99, label='Normal_CVaR_0.99')
        plt.plot(last_year_range, G_student_CVaR_99, label='Student_CVaR_0.99')
        plt.plot(last_year_range, L_act, label='Actual Loss')
        plt.legend()
        plt.xlabel("Weeks")
        plt.suptitle("VaR_0.99 & CVaR_0.99 between two models versus time")
        plt.show()


    def risk_among_all_models(self):
        final_friday_index = np.nonzero(self.weekly_date == '12/29/17')[0][0]
        index_1_2_15 = np.nonzero(self.weekly_date == '1/2/15')[0][0]
        last_year_range = range(0, final_friday_index - index_1_2_15, 4)
        L_act = []

        # GARCH normal
        G_normal_VaR_95 = []
        G_normal_CVaR_95 = []
        G_normal_VaR_99 = []
        G_normal_CVaR_99 = []
        # GARCH student
        G_student_VaR_95 = []
        G_student_CVaR_95 = []
        G_student_VaR_99 = []
        G_student_CVaR_99 = []
        # Loss L
        L_VaR_95 = []
        L_CVaR_95 = []
        L_VaR_99 = []
        L_CVaR_99 = []
        # T-studen loss L_t
        L_t_VaR_95 = []
        L_t_CVaR_95 = []
        L_t_VaR_99 = []
        L_t_CVaR_99 = []
        # Regression estimator
        Yaxis_VaR_regression_95 = []
        Yaxis_VaR_regression_99 = []
        Yaxis_ES_regression_95 = []
        Yaxis_ES_regression_99 = []
        # Calculate GARCH
        for i in last_year_range:
            #if i==53 and i==56:
            #    continue
            print("This is " + str(i))
            L_act.append(-(self.V_t[self.weekly_index[index_1_2_15 + i]] - self.V_t[self.weekly_index[index_1_2_15 + i - 1]]))
            params = self.get_GARCH_mu_sigma_x(i)
            mu_tplus1_normal = params[0][-1]
            sigma_tplus1_normal = np.sqrt(params[1][-1])
            mu_tplus1_student = params[3][-1]
            sigma_tplus1_student = np.sqrt(params[4][-1])
            df = self.get_ARMA_GARCH_parameters(i, "StudentsT")[3][3]

            G_normal_VaR_95.append(mu_tplus1_normal + sigma_tplus1_normal*norm.ppf(0.95))
            G_normal_CVaR_95.append(mu_tplus1_normal + sigma_tplus1_normal*(norm.pdf(norm.ppf(0.95)) / (1 - 0.95)))
            G_normal_VaR_99.append(mu_tplus1_normal + sigma_tplus1_normal * norm.ppf(0.99))
            G_normal_CVaR_99.append(mu_tplus1_normal + sigma_tplus1_normal * (norm.pdf(norm.ppf(0.99)) / (1 - 0.99)))

            G_student_VaR_95.append(mu_tplus1_student + sigma_tplus1_student * t.ppf(0.95, df))
            G_student_CVaR_95.append(mu_tplus1_student + sigma_tplus1_student
                                     * ((t.pdf(t.ppf(0.95, df), df) / (1 - 0.95))) * (((df + (t.ppf(0.95, df))**2) / (df - 1))))
            G_student_VaR_99.append(mu_tplus1_student + sigma_tplus1_student * t.ppf(0.99, df))
            G_student_CVaR_99.append(mu_tplus1_student + sigma_tplus1_student
                                     * ((t.pdf(t.ppf(0.99, df), df) / (1 - 0.99))) * (((df + (t.ppf(0.99, df))**2) / (df - 1))))
        # Calculate Regression
        for i in last_year_range:
            Yaxis_VaR_regression_95.append(self.regression_estimator(i, "No")[1])
            Yaxis_VaR_regression_99.append(self.regression_estimator(i, "No")[2])
            Yaxis_ES_regression_95.append(self.regression_estimator(i, "No")[3])
            Yaxis_ES_regression_99.append(self.regression_estimator(i, "No")[4])
        # Calculate normal and t-student
        for i in range(36, 71):
            print("It's time " + str(i))
            L, L_delta, L_t = self.calibrate(i)
            # Normal case
            # For Loss L
            L_fit_mu, L_fit_sigma = norm.fit(L)
            # Calculate VaR and CVaR
            L_VaR_95.append(norm.ppf(0.95, L_fit_mu, L_fit_sigma))
            L_CVaR_95.append(L_fit_mu + L_fit_sigma * (norm.pdf(norm.ppf(0.95)) / (1 - 0.95)))
            L_VaR_99.append(norm.ppf(0.99, L_fit_mu, L_fit_sigma))
            L_CVaR_99.append(L_fit_mu + L_fit_sigma * (norm.pdf(norm.ppf(0.99)) / (1 - 0.99)))

            # T-student case
            # For Loss L
            while True:
                L_t_fit_df, L_t_fit_mu, L_t_fit_sigma = t.fit(L_t)
                print(L_t_fit_df)
                if L_t_fit_df > 2:
                    break
                L, L_delta, L_t = self.calibrate(i)
            # Calculate VaR and CVaR
            L_t_VaR_95.append(t.ppf(0.95, L_t_fit_df, L_t_fit_mu, L_t_fit_sigma))
            L_t_CVaR_95.append(L_t_fit_mu + L_t_fit_sigma * ((t.pdf(t.ppf(0.95, L_t_fit_df), L_t_fit_df) / (1 - 0.95))) * (
            ((L_t_fit_df + (t.ppf(0.95, L_t_fit_df)) ** 2) / (L_t_fit_df - 1))))
            L_t_VaR_99.append(t.ppf(0.99, L_t_fit_df, L_t_fit_mu, L_t_fit_sigma))
            L_t_CVaR_99.append(L_t_fit_mu + L_t_fit_sigma * ((t.pdf(t.ppf(0.99, L_t_fit_df), L_t_fit_df) / (1 - 0.99))) * (
            ((L_t_fit_df + (t.ppf(0.99, L_t_fit_df)) ** 2) / (L_t_fit_df - 1))))
        # Save lists
        with open("all_models_data_2.txt", "wb") as fp:  # Pickling
            pickle.dump([[G_normal_VaR_95, G_student_VaR_95, L_VaR_95, L_t_VaR_95, Yaxis_VaR_regression_95],
                         [G_normal_CVaR_95, G_student_CVaR_95, L_CVaR_95, L_t_CVaR_95, Yaxis_ES_regression_95],
                         [G_normal_VaR_99, G_student_VaR_99, L_VaR_99, L_t_VaR_99, Yaxis_VaR_regression_99],
                         [G_normal_CVaR_99, G_student_CVaR_99, L_CVaR_99, L_t_CVaR_99, Yaxis_ES_regression_99]], fp)

        # Plot
        # VaR 0.95 case
        plt.plot(last_year_range[0:-4], G_normal_VaR_95[0:-4], label='Garch_Normal_VaR_0.95')
        plt.plot(last_year_range[0:-4], G_student_VaR_95[0:-4], label='Garch_Student_VaR_0.95')
        plt.plot(last_year_range[0:-4], L_VaR_95, label='Normal_dist_VaR_0.95')
        plt.plot(last_year_range[0:-4], L_t_VaR_95, label='t_dist_VaR_0.95')
        plt.plot(last_year_range[0:-4], Yaxis_VaR_regression_95[0:-4], label='Regression_VaR_0.95')
        plt.plot(last_year_range[0:-4], L_act[0:-4], label='Actual Loss')
        plt.legend()
        plt.xlabel("Weeks")
        plt.ylabel("Values")
        plt.title("VaR_0.95 among five models versus time")
        plt.show()
        # CVaR 0.95 case
        plt.plot(last_year_range[0:-4], G_normal_CVaR_95[0:-4], label='Garch_Normal_CVaR_0.95')
        plt.plot(last_year_range[0:-4], G_student_CVaR_95[0:-4], label='Garch_Student_CVaR_0.95')
        plt.plot(last_year_range[0:-4], L_CVaR_95, label='Normal_dist_CVaR_0.95')
        plt.plot(last_year_range[0:-4], L_t_CVaR_95, label='t_dist_CVaR_0.95')
        plt.plot(last_year_range[0:-4], Yaxis_ES_regression_95[0:-4], label='Regression_CVaR_0.95')
        plt.plot(last_year_range[0:-4], L_act[0:-4], label='Actual Loss')
        plt.legend()
        plt.xlabel("Weeks")
        plt.ylabel("Values")
        plt.title("CVaR_0.95 among five models versus time")
        plt.show()
        # VaR 0.99 case
        plt.plot(last_year_range[0:-4], G_normal_VaR_99[0:-4], label='Garch_Normal_VaR_0.99')
        plt.plot(last_year_range[0:-4], G_student_VaR_99[0:-4], label='Garch_Student_VaR_0.99')
        plt.plot(last_year_range[0:-4], L_VaR_99, label='Normal_dist_VaR_0.99')
        plt.plot(last_year_range[0:-4], L_t_VaR_99, label='t_dist_VaR_0.99')
        plt.plot(last_year_range[0:-4], Yaxis_VaR_regression_99[0:-4], label='Regression_VaR_0.99')
        plt.plot(last_year_range[0:-4], L_act[0:-4], label='Actual Loss')
        plt.legend()
        plt.xlabel("Weeks")
        plt.ylabel("Values")
        plt.title("VaR_0.99 among five models versus time")
        plt.show()
        # CVaR 0.99 case
        plt.plot(last_year_range[0:-4], G_normal_CVaR_99[0:-4], label='Garch_Normal_CVaR_0.99')
        plt.plot(last_year_range[0:-4], G_student_CVaR_99[0:-4], label='Garch_Student_CVaR_0.99')
        plt.plot(last_year_range[0:-4], L_CVaR_99, label='Normal_dist_CVaR_0.99')
        plt.plot(last_year_range[0:-4], L_t_CVaR_99, label='t_dist_CVaR_0.99')
        plt.plot(last_year_range[0:-4], Yaxis_ES_regression_99[0:-4], label='Regression_CVaR_0.99')
        plt.plot(last_year_range[0:-4], L_act[0:-4], label='Actual Loss')
        plt.legend()
        plt.xlabel("Weeks")
        plt.ylabel("Values")
        plt.title("CVaR_0.99 among five models versus time")
        plt.show()

        ## Difference between actual monthly losses and VaR & CVaR values
        # VaR.95
        plt.subplot(321)
        plt.plot(last_year_range[0:-4], (np.asarray(L_VaR_95) - L_act[0:-4]) / np.asarray(L_VaR_95),
                 label='Normal_dist_VaR_0.95')
        plt.legend()
        plt.subplot(322)
        plt.plot(last_year_range[0:-4], (np.asarray(L_t_VaR_95) - L_act[0:-4]) / np.asarray(L_t_VaR_95),
                 label='t_dist_VaR_0.95')
        plt.legend()
        plt.subplot(323)
        plt.plot(last_year_range[0:-4], ((np.asarray(Yaxis_VaR_regression_95) - L_act) / np.asarray(Yaxis_VaR_regression_95))[0:-4],
                 label='Regression_VaR_0.95')
        plt.legend()
        plt.subplot(324)
        plt.plot(last_year_range[0:-4],
                 ((np.asarray(G_normal_VaR_95) - L_act) / np.asarray(G_normal_VaR_95))[0:-4],
                 label='GARCH_Normal_VaR_0.95')
        plt.legend()
        plt.subplot(325)
        plt.plot(last_year_range[0:-4],
                 ((np.asarray(G_student_VaR_95) - L_act) / np.asarray(G_student_VaR_95))[0:-4],
                 label='GARCH_Student_VaR_0.95')
        plt.legend()
        #plt.ylim(-1, 10)
        plt.xlabel("Weeks")
        plt.ylabel('percentage')
        plt.show()

        # VaR_99
        plt.subplot(321)
        plt.plot(last_year_range[0:-4], (np.asarray(L_VaR_99) - L_act[0:-4]) / np.asarray(L_VaR_99),
                 label='Normal_dist_VaR_0.99')
        plt.legend()
        plt.subplot(322)
        plt.plot(last_year_range[0:-4], (np.asarray(L_t_VaR_99) - L_act[0:-4]) / np.asarray(L_t_VaR_99),
                 label='t_dist_VaR_0.99')
        plt.legend()
        plt.subplot(323)
        plt.plot(last_year_range[0:-4],
                 ((np.asarray(Yaxis_VaR_regression_99) - L_act) / np.asarray(Yaxis_VaR_regression_99))[0:-4],
                 label='Regression_VaR_0.99')
        plt.legend()
        plt.subplot(324)
        plt.plot(last_year_range[0:-4],
                 ((np.asarray(G_normal_VaR_99) - L_act) / np.asarray(G_normal_VaR_99))[0:-4],
                 label='GARCH_Normal_VaR_0.99')
        plt.legend()
        plt.subplot(325)
        plt.plot(last_year_range[0:-4],
                 ((np.asarray(G_student_VaR_99) - L_act) / np.asarray(G_student_VaR_99))[0:-4],
                 label='GARCH_Student_VaR_0.99')
        plt.legend()
        # plt.ylim(-1, 10)
        plt.xlabel("Weeks")
        plt.ylabel('percentage')
        plt.show()

        # ES.95
        plt.subplot(321)
        plt.plot(last_year_range[0:-4], (np.asarray(L_CVaR_95) - L_act[0:-4]) / np.asarray(L_CVaR_95),
                 label='Normal_dist_CVaR_0.95')
        plt.legend()
        plt.subplot(322)
        plt.plot(last_year_range[0:-4], (np.asarray(L_t_CVaR_95) - L_act[0:-4]) / np.asarray(L_t_CVaR_95),
                 label='t_dist_CVaR_0.95')
        plt.legend()
        plt.subplot(323)
        plt.plot(last_year_range[0:-4],
                 ((np.asarray(Yaxis_ES_regression_95) - L_act) / np.asarray(Yaxis_ES_regression_95))[0:-4],
                 label='Regression_CVaR_0.95')
        plt.legend()
        plt.subplot(324)
        plt.plot(last_year_range[0:-4],
                 ((np.asarray(G_normal_CVaR_95) - L_act) / np.asarray(G_normal_CVaR_95))[0:-4],
                 label='GARCH_Normal_CVaR_0.95')
        plt.legend()
        plt.subplot(325)
        plt.plot(last_year_range[0:-4],
                 ((np.asarray(G_student_CVaR_95) - L_act) / np.asarray(G_student_CVaR_95))[0:-4],
                 label='GARCH_Student_CVaR_0.95')
        plt.legend()
        # plt.ylim(-1, 10)
        plt.xlabel("Weeks")
        plt.ylabel('percentage')
        plt.show()

        # ES.99
        plt.subplot(321)
        plt.plot(last_year_range[0:-4], (np.asarray(L_CVaR_99) - L_act[0:-4]) / np.asarray(L_CVaR_99),
                 label='Normal_dist_CVaR_0.99')
        plt.legend()
        plt.subplot(322)
        plt.plot(last_year_range[0:-4], (np.asarray(L_t_CVaR_99) - L_act[0:-4]) / np.asarray(L_t_CVaR_99),
                 label='t_dist_CVaR_0.99')
        plt.legend()
        plt.subplot(323)
        plt.plot(last_year_range[0:-4],
                 ((np.asarray(Yaxis_ES_regression_99) - L_act) / np.asarray(Yaxis_ES_regression_99))[0:-4],
                 label='Regression_CVaR_0.99')
        plt.legend()
        plt.subplot(324)
        plt.plot(last_year_range[0:-4],
                 ((np.asarray(G_normal_CVaR_99) - L_act) / np.asarray(G_normal_CVaR_99))[0:-4],
                 label='GARCH_Normal_CVaR_0.99')
        plt.legend()
        plt.subplot(325)
        plt.plot(last_year_range[0:-4],
                 ((np.asarray(G_student_CVaR_99) - L_act) / np.asarray(G_student_CVaR_99))[0:-4],
                 label='GARCH_Student_CVaR_0.99')
        plt.legend()
        # plt.ylim(-1, 10)
        plt.xlabel("Weeks")
        plt.ylabel('percentage')
        plt.show()

























solver = Portfolio_Risk(450000,"AAL.csv", "AAPL.csv", "AEO.csv", "AMZN.csv", "CAR.csv",
                        "DLTR.csv", "GOOG.csv", "KNX.csv", "MCD.csv", "NFLX.csv",
                        "NKE.csv", "NVDA.csv", "SINA.csv", "TSLA.csv", "UPS.csv", "XEO_price.csv")
solver.get_price_list()
solver.get_bond_list('GOVPIT_3_1.csv', 'GOVPIT_4_1.csv')
solver.get_option_list('KNX_call50.csv', 'NFLX_call315.csv', 'XEO.csv', 'Greek.csv', 'KNX_vol.csv', 'NFLX_vol.csv',
                       'XEO_vol.csv', 'MMR.csv')
#solver.cal_monthly_V_t()
solver.cal_weekly_index()
solver.cal_weekly_bond_index()
solver.cal_weekly_option_index()
#solver.GARCH_test()
#solver.analyze_all_8_weeks()
# solver.calibrate_stock_weekly(0)
# solver.calibrate_bond_weekly(0)
#solver.calibrate_option_weekly(0)



## Part 1
#solver.compare_normal_t()
solver.plot_dist()
#solver.risk_between_normal_student()
#solver.plotV_t(24, -1, monthly)
#solver.calibrate(24)
#solver.back_test()

## Part 2

#print(solver.regression_estimator(10, "Yes"))
#print(solver.hill_estimator(10, "No"))
#solver.compare_regression_hill_a()
#solver.compare_regression_hill_VaR_ES()
#solver.risk_among_5_models()

## Part 3
#solver.compare_GARCH_dists(0)
#solver.risk_between_GARCH()
#solver.risk_among_all_models()

