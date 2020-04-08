def generate_instance(candidate, root=i):
    instance = []
    j = (candidate.get_name()).strip()

    if candidate.transformation != None:
        transformations_applied = []
        try:
            for p in candidate.parents:
                if p.transformation != None:
                    transformations_applied.extend([p.transformation.name])
                else:
                    pass

            transformations_applied.extend([candidate.transformation.name])

            instance.extend([root, j, tuple(transformations_applied),
                             len(candidate.parents), candidate.get_complexity(),
                             candidate.get_number_of_transformations(),
                             candidate.get_number_of_raw_attributes(), candidate.get_transformation_depth()])

        except KeyError:
            for p in candidate.parents:
                if p.transformation != None:
                    transformations_applied.extend([p.transformation.name])
                else:
                    pass

            transformations_applied.extend([candidate.transformation.name])

            instance.extend([root, j, tuple(transformations_applied),
                             len(candidate.parents), candidate.get_complexity(),
                             candidate.get_number_of_transformations(),
                             candidate.get_number_of_raw_attributes(), candidate.get_transformation_depth()])

    else:
        transformations_applied = []
        for p in candidate.parents:
            if p.transformation != None:
                transformations_applied.extend([p.transformation.name])
            else:
                pass

        transformations_applied.extend(['None'])
        instances.extend([root, j, tuple(transformations_applied),
                          len(candidate.parents), candidate.get_complexity(), 0,
                          candidate.get_number_of_raw_attributes(), 0])

    feature_clf = LogisticRegression(penalty='l2', C=1, solver='lbfgs', class_weight='balanced',
                                     max_iter=100000, multi_class='auto')

    parents_clf = LogisticRegression(penalty='l2', C=1, solver='lbfgs', class_weight='balanced',
                                     max_iter=100000, multi_class='auto')

    parents_type = []
    parents_dtype = []
    if len(candidate.parents) >= 1:
        for p in x.parents:
            if (p.get_name()).strip() in features2_build_cat:
                parents_dtype.extend(['categorical'])
            elif (p.get_name()).strip() in features2_build_num:
                parents_dtype.extend(['numerical'])
            else:
                parents_dtype.extend([p.properties['type']])

            if (p.get_name()).strip() == sensitive_feature:
                parents_type.extend(['inadmissible'])
            elif (p.get_name()).strip() in features2_build_cat:
                categorical_transformer = Pipeline(steps=[
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

                preprocessor_pc = ColumnTransformer(
                    transformers=[
                        ('cat', categorical_transformer, [(p.get_name()).strip()])], remainder='passthrough')

                pipeline_pc = Pipeline(steps=[('preprocessor', preprocessor_pc),
                                              ('clf', parents_clf)])

                X_train_pc = X_train.loc[:, [(p.get_name()).strip()]]
                X_test_pc = X_test.loc[:, [(p.get_name()).strip()]]

                pipeline_pc.fit(X_train_pc, np.ravel(y_train))

                y_pred_proba_parent = pipeline_pc.predict_proba(X_test_pc)[:, 1]
                outcome_parent = pipeline_pc.predict(X_test_pc)

                outcome_p_df = pd.DataFrame(data=outcome_parent, columns=['outcome'])
                sensitive_df = pd.DataFrame(data=X_test.loc[:, sensitive_feature].to_numpy(),
                                            columns=[sensitive_feature])
                parent_df_causal = pd.DataFrame(data=X_test.loc[:, [(p.get_name()).strip()]].to_numpy(),
                                                columns=[(p.get_name()).strip()])
                test_p_df_causal = pd.concat([sensitive_df, parent_df_causal, outcome_p_df], axis=1)

                if d_separation(test_p_df_causal, sensitive=sensitive_feature, target='outcome'):
                    parents_type.extend(['admissible'])
                else:
                    parents_type.extend(['inadmissible'])

            else:
                transformed_train_p = p.transform(X_train_t)
                transformed_test_p = p.transform(X_test_t)

                parents_clf.fit(transformed_train_p, np.ravel(y_train))

                y_pred_proba_parent = parents_clf.predict_proba(transformed_test_p)[:, 1]
                outcome_parent = parents_clf.predict(transformed_test_p)

                outcome_p_df = pd.DataFrame(data=outcome_parent, columns=['outcome'])
                sensitive_df = pd.DataFrame(data=X_test.loc[:, sensitive_feature].to_numpy(),
                                            columns=[sensitive_feature])
                parent_df_causal = pd.DataFrame(data=transformed_test_p, columns=[(p.get_name()).strip()])
                test_p_df_causal = pd.concat([sensitive_df, parent_df_causal, outcome_p_df], axis=1)

                if np.unique(transformed_test_p).shape[0] == 1:
                    parents_type.extend(['admissible'])
                elif d_separation(test_p_df_causal, sensitive=sensitive_feature, target='outcome'):
                    parents_type.extend(['admissible'])
                else:
                    parents_type.extend(['inadmissible'])
    else:
        parents_type.extend(['no parents'])
        parents_dtype.extend(['no parents'])

    instance.extend([tuple(parents_type)])
    instance.extend([tuple(parents_dtype)])
    instance.extend([candidate.properties['type']])

    if sensitive_idx != idx:
        feature_clf.fit(candidate.transform(X_train_t), np.ravel(y_train))
        y_pred_proba_candidate = feature_clf.predict_proba(candidate.transform(X_test_t))[:, 1]
        outcome_candidate = feature_clf.predict(candidate.transform(X_test_t))

        outcome_df = pd.DataFrame(data=outcome_candidate, columns=['outcome'])
        sensitive_df = pd.DataFrame(data=X_test.loc[:, sensitive_feature].to_numpy(),
                                    columns=[sensitive_feature])
        selected_df_causal = pd.DataFrame(data=candidate.transform(X_test_t), columns=[j])
        test_df_causal = pd.concat([sensitive_df, selected_df_causal, outcome_df], axis=1)

        if np.unique(candidate.transform(X_test_t)).shape[0] == 1:
            instance.extend([1])
        elif d_separation(test_df_causal, sensitive=sensitive_feature, target='outcome'):

            instance.extend([1])
        else:
            instance.extend([0])
    else:
        instance.extend(['NA'])

    return instance