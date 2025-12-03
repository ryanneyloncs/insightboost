"""
Visualization suggestion engine for InsightBoost.

This module provides AI-powered visualization recommendations
based on data characteristics and analysis objectives.
"""

from typing import Any, Union
from uuid import uuid4

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from insightboost.api.anthropic_client import AnthropicClient
from insightboost.config.logging_config import get_logger
from insightboost.core.data_analyzer import DataAnalyzer
from insightboost.models.visualization import (
    ChartType,
    GeneratedVisualization,
    VisualizationConfig,
    VisualizationSuggestion,
)
from insightboost.utils.exceptions import VisualizationError
from insightboost.utils.validators import validate_dataframe


logger = get_logger("visualization_suggester")


# Mapping of chart types to plotly express functions
CHART_TYPE_FUNCTIONS = {
    ChartType.SCATTER: px.scatter,
    ChartType.LINE: px.line,
    ChartType.BAR: px.bar,
    ChartType.HISTOGRAM: px.histogram,
    ChartType.BOX: px.box,
    ChartType.VIOLIN: px.violin,
    ChartType.HEATMAP: px.imshow,
    ChartType.PIE: px.pie,
    ChartType.AREA: px.area,
    ChartType.BUBBLE: px.scatter,  # With size parameter
    ChartType.FUNNEL: px.funnel,
    ChartType.TREEMAP: px.treemap,
    ChartType.SUNBURST: px.sunburst,
    ChartType.DENSITY: px.density_heatmap,
}


class VisualizationSuggester:
    """
    AI-powered visualization recommendation engine.
    
    Uses statistical analysis and LLM reasoning to suggest
    optimal chart types for given data and objectives.
    
    Attributes:
        api_client: Anthropic API client
        data_analyzer: Statistical data analyzer
    """
    
    def __init__(
        self,
        api_client: AnthropicClient | None = None,
    ) -> None:
        """
        Initialize the visualization suggester.
        
        Args:
            api_client: Anthropic API client (created if not provided)
        """
        self.api_client = api_client or AnthropicClient()
        self.data_analyzer = DataAnalyzer()
        
        logger.info("VisualizationSuggester initialized")
    
    def suggest_visualizations(
        self,
        df: pd.DataFrame,
        objective: str | None = None,
        max_suggestions: int = 5,
    ) -> list[VisualizationSuggestion]:
        """
        Recommend visualizations based on data characteristics.
        
        Args:
            df: DataFrame to visualize
            objective: Optional analysis objective
            max_suggestions: Maximum suggestions to return
            
        Returns:
            List of VisualizationSuggestion objects
        """
        logger.info("Generating visualization suggestions")
        
        df = validate_dataframe(df)
        
        try:
            # Get AI suggestions
            ai_suggestions = self.api_client.suggest_visualizations(
                df=df,
                objective=objective or "",
                max_suggestions=max_suggestions,
            )
            
            # Add code snippets to suggestions
            for suggestion in ai_suggestions:
                suggestion.code_snippet = self._generate_code_snippet(
                    df_name="df",
                    config=suggestion.config,
                )
            
            # If AI didn't return enough, add rule-based suggestions
            if len(ai_suggestions) < max_suggestions:
                rule_based = self._get_rule_based_suggestions(
                    df=df,
                    exclude_types=[s.chart_type for s in ai_suggestions],
                )
                ai_suggestions.extend(rule_based[:max_suggestions - len(ai_suggestions)])
            
            logger.info(f"Generated {len(ai_suggestions)} visualization suggestions")
            return ai_suggestions
            
        except Exception as e:
            logger.warning(f"AI suggestions failed, falling back to rules: {e}")
            return self._get_rule_based_suggestions(df, max_count=max_suggestions)
    
    def generate_visualization(
        self,
        df: pd.DataFrame,
        suggestion: VisualizationSuggestion,
        interactive: bool = True,
    ) -> go.Figure:
        """
        Generate a visualization from a suggestion.
        
        Args:
            df: DataFrame to visualize
            suggestion: Visualization suggestion to implement
            interactive: Whether to create interactive Plotly figure
            
        Returns:
            Plotly Figure object
        """
        logger.info(f"Generating {suggestion.chart_type.value} visualization")
        
        config = suggestion.config
        
        try:
            fig = self._create_figure(df, config)
            
            # Apply common styling
            fig.update_layout(
                title=config.title or suggestion.title,
                template=config.theme,
                width=config.width,
                height=config.height,
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to generate visualization: {e}")
            raise VisualizationError(
                message=f"Failed to generate visualization: {str(e)}",
                chart_type=suggestion.chart_type.value,
            )
    
    def _create_figure(
        self,
        df: pd.DataFrame,
        config: VisualizationConfig,
    ) -> go.Figure:
        """Create a Plotly figure from configuration."""
        
        chart_type = config.chart_type
        
        # Build arguments for plotly express
        kwargs: dict[str, Any] = {"data_frame": df}
        
        if config.x_column:
            kwargs["x"] = config.x_column
        if config.y_column:
            kwargs["y"] = config.y_column
        if config.color_column:
            kwargs["color"] = config.color_column
        if config.hover_columns:
            kwargs["hover_data"] = config.hover_columns
        
        # Handle specific chart types
        if chart_type == ChartType.SCATTER:
            if config.size_column:
                kwargs["size"] = config.size_column
            return px.scatter(**kwargs)
        
        elif chart_type == ChartType.LINE:
            return px.line(**kwargs)
        
        elif chart_type == ChartType.BAR:
            return px.bar(**kwargs)
        
        elif chart_type == ChartType.HISTOGRAM:
            # Histogram only needs x or y, not both
            if "y" in kwargs and "x" in kwargs:
                del kwargs["y"]
            return px.histogram(**kwargs)
        
        elif chart_type == ChartType.BOX:
            return px.box(**kwargs)
        
        elif chart_type == ChartType.VIOLIN:
            return px.violin(**kwargs)
        
        elif chart_type == ChartType.HEATMAP:
            # Heatmap needs correlation matrix or pivot table
            if config.x_column and config.y_column:
                pivot = df.pivot_table(
                    values=config.y_column,
                    index=config.x_column,
                    aggfunc="mean",
                )
                return px.imshow(pivot, color_continuous_scale=config.color_scale)
            else:
                # Use correlation matrix
                numeric_df = df.select_dtypes(include=["number"])
                corr = numeric_df.corr()
                return px.imshow(
                    corr,
                    color_continuous_scale=config.color_scale,
                    aspect="auto",
                )
        
        elif chart_type == ChartType.PIE:
            if "x" not in kwargs:
                # Use first categorical column
                cat_cols = df.select_dtypes(include=["object", "category"]).columns
                if len(cat_cols) > 0:
                    kwargs["names"] = cat_cols[0]
            else:
                kwargs["names"] = kwargs.pop("x")
            
            if "y" in kwargs:
                kwargs["values"] = kwargs.pop("y")
            
            return px.pie(**kwargs)
        
        elif chart_type == ChartType.AREA:
            return px.area(**kwargs)
        
        elif chart_type == ChartType.BUBBLE:
            if not config.size_column:
                # Find a numeric column for size
                numeric_cols = df.select_dtypes(include=["number"]).columns
                unused = [c for c in numeric_cols if c not in [config.x_column, config.y_column]]
                if unused:
                    kwargs["size"] = unused[0]
            else:
                kwargs["size"] = config.size_column
            return px.scatter(**kwargs)
        
        elif chart_type == ChartType.DENSITY:
            return px.density_heatmap(**kwargs)
        
        else:
            # Fallback to scatter
            logger.warning(f"Unsupported chart type {chart_type}, using scatter")
            return px.scatter(**kwargs)
    
    def auto_visualize(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> dict[str, go.Figure]:
        """
        Generate a complete visualization suite for the dataset.
        
        Args:
            df: DataFrame to visualize
            columns: Specific columns to focus on (None = all)
            
        Returns:
            Dictionary mapping visualization names to figures
        """
        logger.info("Auto-generating visualization suite")
        
        df = validate_dataframe(df)
        
        if columns:
            df = df[columns]
        
        figures: dict[str, go.Figure] = {}
        
        # Distribution visualizations for numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns[:5]
        for col in numeric_cols:
            try:
                fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                fig.update_layout(template="plotly_white")
                figures[f"distribution_{col}"] = fig
            except Exception as e:
                logger.warning(f"Failed to create histogram for {col}: {e}")
        
        # Box plot for numeric columns with categorical
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(numeric_cols) > 0 and len(cat_cols) > 0:
            try:
                fig = px.box(
                    df,
                    x=cat_cols[0],
                    y=numeric_cols[0],
                    title=f"{numeric_cols[0]} by {cat_cols[0]}",
                )
                fig.update_layout(template="plotly_white")
                figures["box_comparison"] = fig
            except Exception as e:
                logger.warning(f"Failed to create box plot: {e}")
        
        # Scatter plot for top correlation
        if len(numeric_cols) >= 2:
            try:
                correlations = self.data_analyzer.compute_correlations(df[numeric_cols])
                if correlations:
                    top_corr = correlations[0]
                    fig = px.scatter(
                        df,
                        x=top_corr.column1,
                        y=top_corr.column2,
                        title=f"Correlation: {top_corr.column1} vs {top_corr.column2} (r={top_corr.correlation:.2f})",
                        trendline="ols",
                    )
                    fig.update_layout(template="plotly_white")
                    figures["top_correlation"] = fig
            except Exception as e:
                logger.warning(f"Failed to create correlation scatter: {e}")
        
        # Correlation heatmap
        if len(numeric_cols) >= 3:
            try:
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(
                    corr_matrix,
                    title="Correlation Matrix",
                    color_continuous_scale="RdBu_r",
                    aspect="auto",
                )
                fig.update_layout(template="plotly_white")
                figures["correlation_heatmap"] = fig
            except Exception as e:
                logger.warning(f"Failed to create correlation heatmap: {e}")
        
        # Pie chart for categorical with low cardinality
        for col in cat_cols[:2]:
            if df[col].nunique() <= 10:
                try:
                    fig = px.pie(
                        df,
                        names=col,
                        title=f"Distribution of {col}",
                    )
                    fig.update_layout(template="plotly_white")
                    figures[f"pie_{col}"] = fig
                    break  # Only one pie chart
                except Exception as e:
                    logger.warning(f"Failed to create pie chart for {col}: {e}")
        
        logger.info(f"Generated {len(figures)} visualizations")
        return figures
    
    @staticmethod
    def determine_chart_type(
        column_types: dict[str, str],
        cardinality: dict[str, int],
        relationship_type: str,
    ) -> str:
        """
        Rule-based chart type determination.
        
        Args:
            column_types: Dictionary of column names to types
            cardinality: Dictionary of column names to unique counts
            relationship_type: Type of relationship to visualize
            
        Returns:
            Recommended chart type string
        """
        numeric_cols = [c for c, t in column_types.items() if t == "numeric"]
        categorical_cols = [c for c, t in column_types.items() if t == "categorical"]
        datetime_cols = [c for c, t in column_types.items() if t == "datetime"]
        
        # Distribution of single variable
        if relationship_type == "distribution":
            if numeric_cols:
                return "histogram"
            elif categorical_cols:
                return "bar"
        
        # Comparison between categories
        elif relationship_type == "comparison":
            if categorical_cols and numeric_cols:
                return "bar"
            elif len(categorical_cols) >= 2:
                return "heatmap"
        
        # Correlation between variables
        elif relationship_type == "correlation":
            if len(numeric_cols) >= 2:
                return "scatter"
            elif len(numeric_cols) >= 3:
                return "scatter_matrix"
        
        # Time series / trends
        elif relationship_type == "trend":
            if datetime_cols and numeric_cols:
                return "line"
        
        # Composition / proportions
        elif relationship_type == "composition":
            if categorical_cols:
                first_cat = categorical_cols[0]
                if cardinality.get(first_cat, 0) <= 10:
                    return "pie"
                else:
                    return "bar"
        
        # Default to scatter for two numeric, bar otherwise
        if len(numeric_cols) >= 2:
            return "scatter"
        elif categorical_cols:
            return "bar"
        else:
            return "histogram"
    
    def _get_rule_based_suggestions(
        self,
        df: pd.DataFrame,
        exclude_types: list[ChartType] | None = None,
        max_count: int = 5,
    ) -> list[VisualizationSuggestion]:
        """Generate rule-based visualization suggestions."""
        
        exclude_types = exclude_types or []
        suggestions = []
        
        numeric_cols = list(df.select_dtypes(include=["number"]).columns)
        categorical_cols = list(df.select_dtypes(include=["object", "category"]).columns)
        
        # Histogram for numeric distributions
        if ChartType.HISTOGRAM not in exclude_types and numeric_cols:
            col = numeric_cols[0]
            suggestions.append(VisualizationSuggestion(
                chart_type=ChartType.HISTOGRAM,
                title=f"Distribution of {col}",
                description=f"Shows the frequency distribution of {col} values",
                config=VisualizationConfig(
                    chart_type=ChartType.HISTOGRAM,
                    x_column=col,
                    title=f"Distribution of {col}",
                ),
                reasoning="Histograms are ideal for understanding the distribution of numeric variables",
                confidence=0.9,
                code_snippet=self._generate_code_snippet("df", VisualizationConfig(
                    chart_type=ChartType.HISTOGRAM,
                    x_column=col,
                )),
            ))
        
        # Scatter plot for correlations
        if ChartType.SCATTER not in exclude_types and len(numeric_cols) >= 2:
            x_col, y_col = numeric_cols[0], numeric_cols[1]
            suggestions.append(VisualizationSuggestion(
                chart_type=ChartType.SCATTER,
                title=f"{x_col} vs {y_col}",
                description=f"Explore the relationship between {x_col} and {y_col}",
                config=VisualizationConfig(
                    chart_type=ChartType.SCATTER,
                    x_column=x_col,
                    y_column=y_col,
                    title=f"{x_col} vs {y_col}",
                ),
                reasoning="Scatter plots reveal correlations and patterns between two numeric variables",
                confidence=0.85,
                code_snippet=self._generate_code_snippet("df", VisualizationConfig(
                    chart_type=ChartType.SCATTER,
                    x_column=x_col,
                    y_column=y_col,
                )),
            ))
        
        # Bar chart for categorical
        if ChartType.BAR not in exclude_types and categorical_cols:
            col = categorical_cols[0]
            if df[col].nunique() <= 20:
                suggestions.append(VisualizationSuggestion(
                    chart_type=ChartType.BAR,
                    title=f"{col} Value Counts",
                    description=f"Shows the frequency of each {col} category",
                    config=VisualizationConfig(
                        chart_type=ChartType.BAR,
                        x_column=col,
                        title=f"{col} Value Counts",
                    ),
                    reasoning="Bar charts effectively compare categorical values",
                    confidence=0.85,
                    code_snippet=self._generate_code_snippet("df", VisualizationConfig(
                        chart_type=ChartType.BAR,
                        x_column=col,
                    )),
                ))
        
        # Box plot for categorical vs numeric
        if ChartType.BOX not in exclude_types and categorical_cols and numeric_cols:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            if df[cat_col].nunique() <= 10:
                suggestions.append(VisualizationSuggestion(
                    chart_type=ChartType.BOX,
                    title=f"{num_col} by {cat_col}",
                    description=f"Compare {num_col} distributions across {cat_col} categories",
                    config=VisualizationConfig(
                        chart_type=ChartType.BOX,
                        x_column=cat_col,
                        y_column=num_col,
                        title=f"{num_col} by {cat_col}",
                    ),
                    reasoning="Box plots show distribution differences across categories",
                    confidence=0.8,
                    code_snippet=self._generate_code_snippet("df", VisualizationConfig(
                        chart_type=ChartType.BOX,
                        x_column=cat_col,
                        y_column=num_col,
                    )),
                ))
        
        # Heatmap for correlations
        if ChartType.HEATMAP not in exclude_types and len(numeric_cols) >= 3:
            suggestions.append(VisualizationSuggestion(
                chart_type=ChartType.HEATMAP,
                title="Correlation Matrix",
                description="Shows correlations between all numeric variables",
                config=VisualizationConfig(
                    chart_type=ChartType.HEATMAP,
                    title="Correlation Matrix",
                ),
                reasoning="Correlation heatmaps reveal relationships across multiple variables",
                confidence=0.8,
                code_snippet="import plotly.express as px\ncorr = df.select_dtypes(include='number').corr()\nfig = px.imshow(corr, color_continuous_scale='RdBu_r')",
            ))
        
        return suggestions[:max_count]
    
    def _generate_code_snippet(
        self,
        df_name: str,
        config: VisualizationConfig,
    ) -> str:
        """Generate Python code snippet for a visualization."""
        
        lines = ["import plotly.express as px", ""]
        
        chart_type = config.chart_type.value
        
        # Build function call
        func_name = f"px.{chart_type}"
        args = [f"{df_name}"]
        
        if config.x_column:
            args.append(f'x="{config.x_column}"')
        if config.y_column:
            args.append(f'y="{config.y_column}"')
        if config.color_column:
            args.append(f'color="{config.color_column}"')
        if config.size_column:
            args.append(f'size="{config.size_column}"')
        if config.title:
            args.append(f'title="{config.title}"')
        
        args_str = ", ".join(args)
        lines.append(f"fig = {func_name}({args_str})")
        lines.append("fig.show()")
        
        return "\n".join(lines)
    
    def export_figure(
        self,
        fig: go.Figure,
        format: str = "html",
        filename: str | None = None,
    ) -> str | bytes:
        """
        Export a figure to various formats.
        
        Args:
            fig: Plotly figure to export
            format: Export format ('html', 'png', 'svg', 'json')
            filename: Optional filename to save to
            
        Returns:
            Exported content as string or bytes
        """
        if format == "html":
            content = fig.to_html(include_plotlyjs="cdn")
        elif format == "json":
            content = fig.to_json()
        elif format == "png":
            content = fig.to_image(format="png", scale=2)
        elif format == "svg":
            content = fig.to_image(format="svg")
        else:
            raise VisualizationError(f"Unsupported export format: {format}")
        
        if filename:
            mode = "wb" if isinstance(content, bytes) else "w"
            with open(filename, mode) as f:
                f.write(content)
        
        return content
