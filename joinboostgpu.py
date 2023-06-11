import cudf

class node:
    def __init__(self, fact, dim_df, dim_key, dim_feature):
        self.fact = fact
        self.message_storage = dict()
        self.splits = dict()
        self.dim_df = dim_df
        self.dim_key = dim_key
        self.dim_feature = dim_feature
    def compute_dummy(self):
        ts = self.fact.agg({'REVENUE': 'sum'}).iloc[0]
        self.tc = self.fact.agg({'REVENUE': 'count'}).iloc[0]
        self.fact['REVENUE'] -= ts/self.tc
        self.ts = 0
    def assign_total(self,ts, tc):
        self.ts = ts
        self.tc = tc
    def find_best_splits(self):
        for relation in self.dim_key:
            key = self.dim_key[relation]
            self.message_storage[relation] = self.fact.groupby(key).agg({'REVENUE': 'sum', key:'count'})
            self.message_storage[relation].columns=["s", "c"]

        ts, tc = self.ts, self.tc
        absorptions = []

        for relation in self.dim_feature:
            key = self.dim_key[relation]
            absorption = self.dim_df[relation].merge(self.message_storage[relation], on=key)
            absorption = absorption.melt(id_vars=['s', 'c'], value_vars=self.dim_feature[relation], var_name='key', value_name='value')
            absorption["relation"] = relation
            absorptions.append(absorption)

        result = cudf.concat(absorptions)
        result = result.groupby(["relation", "key", "value"]).sum().reset_index()
        result = result.sort_values(["relation", 'key', 'value'])
        result[['s', 'c']] = result.groupby(["relation","key"])[['s', 'c']].cumsum()

        if result['s'].dtype != 'float64':
            result['s'] = result['s'].astype('float64')
        if result['c'].dtype != 'float64':
            result['c'] = result['c'].astype('float64')

        result = result[result['c'] < tc]

        result["ts"] = float(ts)
        result["tc"] = float(tc)

        result = result.reset_index().assign(criteria=result.eval('(s*s/c) + ((ts-s)* (ts - s))/(tc-c)'))
        idx = result.groupby(['relation', 'key'])['criteria'].idxmax()
        result = result.iloc[idx]

        max_row = result.nlargest(1, 'criteria')
        max_value = max_row["criteria"].iloc[-1]
        max_s = max_row["s"].iloc[-1]
        max_c = max_row["c"].iloc[-1]
        max_index = max_row["value"].iloc[-1]
        relation = max_row["relation"].iloc[-1]
        feature = max_row["key"].iloc[-1]
        self.splits[max_value] = (relation, feature, max_index, max_s, max_c)
        print("splitting relation", relation, "feature", feature, "value", max_index)
                
    def split(self):
        max_key = max(self.splits.keys())
        relation, feature, max_index, max_s, max_c = self.splits[max_key]
        df = self.dim_df[relation]
        key = self.dim_key[relation]
        left_ts, left_tc, right_ts, right_tc = 0,0,0,0 
        if max_index > 500:
            msg = df[[feature]][df[feature] > max_index]
            left_ts = self.ts - max_s
            left_tc = self.tc - max_c
            right_ts = max_s
            right_tc = max_c
        else:
            msg = df[[feature]][df[feature] <= max_index]
            left_ts = max_s
            left_tc = max_c
            right_ts = self.ts - max_s
            right_tc = self.tc - max_c
        n1 = node(self.fact.merge(msg, on=key, how='leftsemi'), self.dim_df, self.dim_key, self.dim_feature)

        n1.assign_total(left_ts, left_tc)
        n2 = node(self.fact.merge(msg, on=key, how='leftanti'), self.dim_df, self.dim_key, self.dim_feature)
        n2.assign_total(right_ts, right_tc)

        self.clean()
        return n1, n2
    
    def clean(self):
        del self.fact
        for relation in self.message_storage:
            df = self.message_storage[relation]
            del df
            
def train_decision_tree(fact, dim_df, dim_key, dim_feature):
    n0 = node(fact, dim_df, dim_key, dim_feature)
    n0.compute_dummy()
    n0.find_best_splits()
    n1, n2 = n0.split()
    n1.find_best_splits()
    n2.find_best_splits()
    n3, n4  = n1.split()
    n5, n6  = n2.split()
    n3.find_best_splits()
    n4.find_best_splits()
    n5.find_best_splits()
    n6.find_best_splits()
    l1, l2 = n3.split()
    l3, l4 = n4.split()
    l5, l6 = n5.split()
    l7, l8 = n6.split()
